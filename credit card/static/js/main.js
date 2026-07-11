// Credit Card Approval Prediction System - Modern Banking Client-Side Logic

document.addEventListener('DOMContentLoaded', () => {
    // 1. Preloader Fade Out
    const preloader = document.getElementById('preloader');
    if (preloader) {
        window.addEventListener('load', () => {
            setTimeout(() => {
                preloader.classList.add('fade-out');
            }, 300);
        });
        // Safety timeout in case load event fires early or late
        setTimeout(() => {
            preloader.classList.add('fade-out');
        }, 1500);
    }

    // 2. Dark/Light Theme Switching
    const themeToggleBtn = document.getElementById('theme-toggle');
    const currentTheme = localStorage.getItem('theme') || 'light';
    
    // Set initial theme
    document.documentElement.setAttribute('data-theme', currentTheme);
    updateThemeToggleIcon(currentTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            let activeTheme = document.documentElement.getAttribute('data-theme');
            let newTheme = activeTheme === 'dark' ? 'light' : 'dark';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeToggleIcon(newTheme);
            
            // Re-trigger chart coloring if on analytics dashboard
            if (window.analyticsCharts) {
                window.location.reload(); // Quick reset to update Chart.js colors
            }
        });
    }

    function updateThemeToggleIcon(theme) {
        if (!themeToggleBtn) return;
        const icon = themeToggleBtn.querySelector('i');
        if (theme === 'dark') {
            icon.className = 'bi bi-sun-fill';
        } else {
            icon.className = 'bi bi-moon-stars-fill';
        }
    }

    // 3. Form Validations
    const predictionForm = document.getElementById('creditForm');
    if (predictionForm) {
        // Automatically sync Income and Annual Income if one is typed
        const incomeInput = document.getElementById('income');
        const annualIncomeInput = document.getElementById('annual_income');

        if (incomeInput && annualIncomeInput) {
            incomeInput.addEventListener('input', () => {
                const val = parseFloat(incomeInput.value) || 0;
                if (val > 0 && (!annualIncomeInput.value || annualIncomeInput.value == val * 10)) {
                    annualIncomeInput.value = Math.round(val * 12);
                }
            });
        }

        predictionForm.addEventListener('submit', (e) => {
            const age = parseInt(document.getElementById('age').value);
            const dependents = parseInt(document.getElementById('dependents').value);
            const existingLoans = parseInt(document.getElementById('existing_loans').value);
            const loanAmount = parseFloat(document.getElementById('loan_amount').value);
            const income = parseFloat(document.getElementById('income').value);

            let errors = [];

            if (age < 18 || age > 100) {
                errors.push("Applicant age must be between 18 and 100.");
            }
            if (dependents < 0) {
                errors.push("Number of dependents cannot be negative.");
            }
            if (existingLoans < 0) {
                errors.push("Existing loans count cannot be negative.");
            }
            if (loanAmount <= 0) {
                errors.push("Requested loan amount must be greater than zero.");
            }
            if (income <= 0) {
                errors.push("Monthly income must be greater than zero.");
            }

            if (errors.length > 0) {
                e.preventDefault();
                showToast(errors.join("<br>"), "danger");
            } else {
                // Show loading spinner during submission
                const submitBtn = predictionForm.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing Application...';
                }
            }
        });
    }

    // 4. Animate Credit Score Speedometer Gauge
    const scoreElement = document.getElementById('creditScoreVal');
    if (scoreElement) {
        const score = parseInt(scoreElement.getAttribute('data-score')) || 300;
        
        // Gauge bounds: 300 (0%) to 850 (100%)
        const percentage = Math.min(Math.max((score - 300) / 550, 0), 1);
        
        // Speedometer rotation: -90deg (300 score) to 90deg (850 score)
        const rotationAngle = -90 + (percentage * 180);
        
        setTimeout(() => {
            const gaugeFill = document.querySelector('.gauge-fill');
            const gaugeNeedle = document.querySelector('.gauge-needle');
            
            if (gaugeFill) {
                // Rotate fill boundary
                gaugeFill.style.transform = `rotate(${45 + (percentage * 180)}deg)`;
            }
            if (gaugeNeedle) {
                gaugeNeedle.style.transform = `translateX(-50%) rotate(${rotationAngle}deg)`;
            }
            
            // Count up animation for score text
            let currentVal = 300;
            const duration = 1500; // 1.5 seconds
            const stepTime = Math.abs(Math.floor(duration / (score - 300)));
            
            const timer = setInterval(() => {
                if (currentVal >= score) {
                    scoreElement.innerText = score;
                    clearInterval(timer);
                } else {
                    currentVal += 5;
                    if (currentVal > score) currentVal = score;
                    scoreElement.innerText = currentVal;
                }
            }, Math.max(stepTime, 10));
        }, 300);
    }

    // 5. Toast Notifications Helper
    window.showToast = function(message, type = "info") {
        const toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            // Create container if it doesn't exist
            const container = document.createElement('div');
            container.id = 'toastContainer';
            container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        
        const toastId = 'toast_' + Date.now();
        const iconMap = {
            'success': 'bi-check-circle-fill',
            'danger': 'bi-exclamation-triangle-fill',
            'warning': 'bi-exclamation-circle-fill',
            'info': 'bi-info-circle-fill'
        };
        
        const toastHtml = `
            <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true" data-bs-delay="5000">
                <div class="d-flex">
                    <div class="toast-body d-flex align-items-center gap-2">
                        <i class="bi ${iconMap[type] || 'bi-info-circle-fill'}"></i>
                        <div>${message}</div>
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            </div>
        `;
        
        document.getElementById('toastContainer').insertAdjacentHTML('beforeend', toastHtml);
        const toastEl = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastEl);
        toast.show();
        
        toastEl.addEventListener('hidden.bs.toast', () => {
            toastEl.remove();
        });
    };
});
