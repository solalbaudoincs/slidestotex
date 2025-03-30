document.addEventListener('DOMContentLoaded', function() {
    const gridBtn = document.getElementById('grid-view-btn');
    const listBtn = document.getElementById('list-view-btn');
    const container = document.getElementById('preview-container');
    
    // Check if there's a saved preference
    const viewMode = localStorage.getItem('slidestotex-view-mode') || 'grid';
    setViewMode(viewMode);
    
    gridBtn.addEventListener('click', function() {
        setViewMode('grid');
    });
    
    listBtn.addEventListener('click', function() {
        setViewMode('list');
    });
    
    function setViewMode(mode) {
        if (mode === 'grid') {
            container.className = 'preview-grid';
            gridBtn.classList.add('active');
            listBtn.classList.remove('active');
        } else {
            container.className = 'preview-list';
            listBtn.classList.add('active');
            gridBtn.classList.remove('active');
        }
        localStorage.setItem('slidestotex-view-mode', mode);
    }
});
