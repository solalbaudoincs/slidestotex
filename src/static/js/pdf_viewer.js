document.addEventListener('DOMContentLoaded', function() {
    // Get references to important DOM elements
    const pages = document.querySelectorAll('.pdf-page');
    const latexSections = document.querySelectorAll('.latex-section');
    const leftPanel = document.querySelector('.left-panel');
    const darkModeToggle = document.getElementById('darkModeToggle');
    const lastPageIndex = localStorage.getItem('lastPageIndex') || 0;
    
    // Check for saved dark mode preference
    if (localStorage.getItem('darkMode') === 'enabled') {
        document.body.classList.add('dark-mode');
        darkModeToggle.textContent = 'Mode Clair';
    }
    
    // Dark mode toggle handler
    darkModeToggle.addEventListener('click', function() {
        document.body.classList.toggle('dark-mode');
        
        if (document.body.classList.contains('dark-mode')) {
            localStorage.setItem('darkMode', 'enabled');
            darkModeToggle.textContent = 'Light Mode';
        } else {
            localStorage.setItem('darkMode', 'disabled');
            darkModeToggle.textContent = 'Dark Mode';
        }
    });
    
    // Function to update LaTeX display based on scroll position
    function updateLatexDisplay() {
        const screenCenter = leftPanel.scrollTop + leftPanel.clientHeight / 2;
        let closestPage = null;
        let minDistance = Infinity;
        
        // Find the page closest to the center of the viewport
        pages.forEach(page => {
            const rect = page.getBoundingClientRect();
            const pageCenter = page.offsetTop + (rect.height / 2);
            const distance = Math.abs(pageCenter - screenCenter);
            
            if (distance < minDistance) {
                minDistance = distance;
                closestPage = page;
            }
        });
        
        if (closestPage) {
            latexSections.forEach(el => el.style.display = 'none');
            const idx = closestPage.dataset.index;
            const targetLatex = document.getElementById('latex-' + idx);
            if (targetLatex) targetLatex.style.display = 'block';
            localStorage.setItem('lastPageIndex', idx);
        }
    }
    
    leftPanel.addEventListener('scroll', updateLatexDisplay);
    //focus first page on load
    const lastPage = document.querySelector('.pdf-page[data-index="' + lastPageIndex + '"]');
    setTimeout(() => {
        if (lastPage) {
            console.log('Restoring last page index:', lastPageIndex);
            lastPage.scrollIntoView({ behavior: 'smooth' });
            lastPage.classList.add('active-page');
        }
    }, 1000);
});
