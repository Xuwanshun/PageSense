"""Evaluation framework for the manual/operation-handbook RAG system.

Design rule: the *default* code path is fully offline. It scores a recorded
``predictions.json`` against a gold set using pure-Python metrics — no Paddle,
no OpenAI, no AWS/S3/ECS. Anything that calls a model (regenerating predictions
or LLM-as-judge) lives behind explicit ``--live`` / ``--confirm-cloud-cost``
flags. See ``eval/README.md``.
"""
