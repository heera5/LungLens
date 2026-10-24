/**
 * LungLens - Clean Human-Centered Interactive Utilities
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Age Slider & Number Input Synchronization
  const ageSlider = document.getElementById('ageRange');
  const ageInput = document.getElementById('ageInput');

  if (ageSlider && ageInput) {
    ageSlider.addEventListener('input', (e) => {
      ageInput.value = e.target.value;
    });

    ageInput.addEventListener('input', (e) => {
      let val = parseInt(e.target.value) || 18;
      if (val < 1) val = 1;
      if (val > 100) val = 100;
      ageSlider.value = val;
    });
  }

  // 2. Toggle Password Visibility (Text & Icon Support)
  const togglePassBtns = document.querySelectorAll('.toggle-password');
  togglePassBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target') || 'passwordInput';
      const input = document.getElementById(targetId);
      if (input) {
        const isPassword = input.getAttribute('type') === 'password';
        input.setAttribute('type', isPassword ? 'text' : 'password');
        btn.textContent = isPassword ? 'Hide' : 'Show';
      }
    });
  });

  // 3. Quick-fill Demo Doctor Account
  const demoFillBtn = document.getElementById('demoFillBtn');
  if (demoFillBtn) {
    demoFillBtn.addEventListener('click', (e) => {
      e.preventDefault();
      const userField = document.getElementById('usernameInput');
      const passField = document.getElementById('passwordInput');
      if (userField && passField) {
        userField.value = 'doctor';
        passField.value = 'doctor123';
      }
    });
  }

  // 4. Auto-dismiss Alert Messages
  const alerts = document.querySelectorAll('.alert');
  alerts.forEach((alert) => {
    const closeBtn = alert.querySelector('.alert-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        alert.style.transition = 'opacity 0.25s ease';
        alert.style.opacity = '0';
        setTimeout(() => alert.remove(), 250);
      });
    }

    setTimeout(() => {
      if (alert.parentElement) {
        alert.style.transition = 'opacity 0.4s ease';
        alert.style.opacity = '0';
        setTimeout(() => alert.remove(), 400);
      }
    }, 5000);
  });
});

/**
 * Load Sample Patient Profile for Testing
 * @param {'low'|'high'} profile
 */
function loadPreset(profile) {
  const presets = {
    low: {
      gender: '0', // Female
      age: 32,
      smoking: '0',
      yellow_fingers: '0',
      anxiety: '0',
      peer_pressure: '0',
      chronic_disease: '0',
      fatigue: '0',
      allergy: '0',
      wheezing: '0',
      alcohol_consuming: '0',
      coughing: '0',
      shortness_of_breath: '0',
      swallowing_difficulty: '0',
      chest_pain: '0'
    },
    high: {
      gender: '1', // Male
      age: 65,
      smoking: '1',
      yellow_fingers: '1',
      anxiety: '1',
      peer_pressure: '1',
      chronic_disease: '1',
      fatigue: '0',
      allergy: '1',
      wheezing: '1',
      alcohol_consuming: '1',
      coughing: '0',
      shortness_of_breath: '0',
      swallowing_difficulty: '1',
      chest_pain: '0'
    }
  };

  const data = presets[profile];
  if (!data) return;

  // Set biological sex
  const genderRadios = document.querySelectorAll('input[name="gender"]');
  genderRadios.forEach((r) => {
    r.checked = (r.value === data.gender);
  });

  // Set age
  const ageSlider = document.getElementById('ageRange');
  const ageInput = document.getElementById('ageInput');
  if (ageSlider && ageInput) {
    ageSlider.value = data.age;
    ageInput.value = data.age;
  }

  // Set binary radio buttons
  const keys = [
    'smoking', 'yellow_fingers', 'anxiety', 'peer_pressure',
    'chronic_disease', 'fatigue', 'allergy', 'wheezing',
    'alcohol_consuming', 'coughing', 'shortness_of_breath',
    'swallowing_difficulty', 'chest_pain'
  ];

  keys.forEach((key) => {
    const radios = document.querySelectorAll(`input[name="${key}"]`);
    radios.forEach((r) => {
      r.checked = (r.value === data[key]);
    });
  });
}

/**
 * Clear Form to Neutral Defaults
 */
function resetAssessmentForm() {
  const form = document.getElementById('assessmentForm');
  if (form) {
    form.reset();
    const ageSlider = document.getElementById('ageRange');
    const ageInput = document.getElementById('ageInput');
    if (ageSlider && ageInput) {
      ageSlider.value = 45;
      ageInput.value = 45;
    }
    const noRadios = form.querySelectorAll('.opt-no input[type="radio"]');
    noRadios.forEach((r) => {
      r.checked = true;
    });
  }
}
