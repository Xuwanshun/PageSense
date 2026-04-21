const tabs = document.querySelectorAll('.auth-tab');
const submitBtn = document.getElementById('submit-btn');
const authForm = document.getElementById('auth-form');
const errorEl = document.getElementById('auth-error');
let mode = 'signin';

tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    mode = tab.dataset.tab;
    tabs.forEach((t) => t.classList.toggle('active', t === tab));
    submitBtn.textContent = mode === 'signin' ? 'Sign In' : 'Create Account';
    document.getElementById('password').autocomplete =
      mode === 'signin' ? 'current-password' : 'new-password';
    hideError();
  });
});

// Consume access token dropped in URL fragment by OAuth callback
const fragment = new URLSearchParams(window.location.hash.slice(1));
const fragmentToken = fragment.get('token');
if (fragmentToken) {
  history.replaceState(null, '', window.location.pathname);
  sessionStorage.setItem('__auth_token_once', fragmentToken);
  window.location.href = '/';
}

authForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideError();
  const email = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value;
  const endpoint = mode === 'signin' ? '/auth/login' : '/auth/register';

  submitBtn.disabled = true;
  submitBtn.textContent = '…';
  try {
    const r = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await r.json();
    if (!r.ok) {
      showError(data.detail || 'Authentication failed.');
      return;
    }
    sessionStorage.setItem('__auth_token_once', data.access_token);
    window.location.href = '/';
  } catch {
    showError('Network error. Is the server running?');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = mode === 'signin' ? 'Sign In' : 'Create Account';
  }
});

function showError(msg) { errorEl.textContent = msg; errorEl.hidden = false; }
function hideError() { errorEl.hidden = true; }
