/**
 * Copies text from the parent's `data-copy` attribute or from a target element's text.
 * @param {HTMLElement} btn - Clicked copy icon element.
 */
function copyToClipboard(btn) {
    const targetId = btn.getAttribute('data-copy-target');
    let textToCopy = '';

    if (targetId) {
        const targetElement = document.getElementById(targetId);
        textToCopy = targetElement ? targetElement.innerText : '';
    } else {
        textToCopy = btn.parentElement.getAttribute('data-copy');
    }

    if (!textToCopy) return;

    navigator.clipboard.writeText(textToCopy).then(() => {
        const icon = btn.querySelector('i');
        if (icon) {
            icon.className = 'fa-solid fa-check';
            setTimeout(() => icon.className = 'fa-regular fa-copy', 2000);
        }
    });
}
