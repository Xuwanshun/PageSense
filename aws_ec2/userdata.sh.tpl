#!/bin/bash
# EC2 userdata: installs Python deps, syncs code/data from S3.
# Training is NOT auto-started — SSH in and run manually.
# The CloudWatch alarm in main.tf stops the instance when CPU goes idle.
set -euo pipefail
exec > /var/log/userdata.log 2>&1

export DEBIAN_FRONTEND=noninteractive
export AWS_DEFAULT_REGION="${aws_region}"

# ── System packages ────────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y git wget curl unzip awscli python3-venv python3-pip

# ── Project directory ──────────────────────────────────────────────────────────
mkdir -p /home/ubuntu/training
chown ubuntu:ubuntu /home/ubuntu/training
cd /home/ubuntu/training

# ── Python virtual environment ─────────────────────────────────────────────────
sudo -u ubuntu python3 -m venv venv
sudo -u ubuntu /home/ubuntu/training/venv/bin/pip install --upgrade pip

# ── PyTorch (CUDA 12.4) ────────────────────────────────────────────────────────
sudo -u ubuntu /home/ubuntu/training/venv/bin/pip install \
  torch==2.6.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

# ── Training dependencies ──────────────────────────────────────────────────────
# transformers dev build required until Qwen3VLForConditionalGeneration ships in stable
sudo -u ubuntu /home/ubuntu/training/venv/bin/pip install \
  "git+https://github.com/huggingface/transformers@main" \
  "peft>=0.17.1" \
  "bitsandbytes>=0.43.0" \
  "accelerate>=0.34.0" \
  "datasets>=2.18.0" \
  "Pillow" \
  "huggingface_hub"

# flash-attn (optional — improves throughput ~20 %; skip if build fails)
sudo -u ubuntu /home/ubuntu/training/venv/bin/pip install flash-attn --no-build-isolation || \
  echo "flash-attn build failed — using sdpa attention (set attn_implementation=sdpa in args)"

# ── HuggingFace authentication ────────────────────────────────────────────────
# Prefer fetching the token from Secrets Manager (avoids embedding it in userdata).
# Store with: aws secretsmanager create-secret --name hf-token --secret-string "<token>"
# Then set hf_token="" in terraform.tfvars and pass the ARN as hf_token_secret_arn.
%{ if hf_token != "" }
sudo -u ubuntu /home/ubuntu/training/venv/bin/python3 -c \
  "from huggingface_hub import login; login('${hf_token}')"
%{ endif }

# ── Sync code and dataset from S3 ────────────────────────────────────────────
# Upload your code first: aws s3 sync . s3://${s3_bucket}/code/ --exclude '.git/*'
aws s3 sync s3://${s3_bucket}/code/    /home/ubuntu/training/ --region ${aws_region} || true
aws s3 sync s3://${s3_bucket}/dataset/ /home/ubuntu/training/dataset/ --region ${aws_region} || true

# ── Self-stop helper script ───────────────────────────────────────────────────
# Call this at the end of your training run to stop the instance immediately:
#   /home/ubuntu/training/stop_self.sh
cat > /home/ubuntu/training/stop_self.sh << 'STOP_SCRIPT'
#!/bin/bash
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
echo "Stopping instance $INSTANCE_ID in $REGION ..."
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
STOP_SCRIPT
chmod +x /home/ubuntu/training/stop_self.sh

# ── Launch script template ────────────────────────────────────────────────────
cat > /home/ubuntu/training/run_training.sh << RUN_SCRIPT
#!/bin/bash
set -euo pipefail
cd /home/ubuntu/training
source venv/bin/activate

# Bucket injected at instance launch by Terraform — no discovery needed.
BUCKET="${s3_bucket}"
REGION="${aws_region}"

python train.py \
  --data_path dataset/train.json \
  --output_dir output/qwen3-vl-sft \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --bf16 True \
  --gradient_checkpointing True \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --save_steps 100 \
  --save_total_limit 3 \
  --logging_steps 10 \
  --eval_strategy no \
  --dataloader_num_workers 2

# Upload final checkpoint to S3
aws s3 sync output/qwen3-vl-sft "s3://\$BUCKET/checkpoints/" --region "\$REGION"

# Stop the instance to save cost
/home/ubuntu/training/stop_self.sh
RUN_SCRIPT
chmod +x /home/ubuntu/training/run_training.sh

chown -R ubuntu:ubuntu /home/ubuntu/training

echo "=== Setup complete ==="
echo "SSH in and run:  cd ~/training && bash run_training.sh"
echo "Or start tmux and run in background: tmux new-session -d -s train 'bash run_training.sh'"
