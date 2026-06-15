"use strict";

/**
 * Copy text to clipboard (handles both code snippets and install commands).
 */
function homepageCopyToClipboard(btn) {
    var targetId = btn.getAttribute("data-copy-target");
    var el = targetId ? document.getElementById(targetId) : null;
    var text = el ? el.innerText : btn.parentElement.getAttribute("data-copy");

    if (!text) return;

    var showSuccess = function () {
        var icon = btn.querySelector("i");
        if (icon) icon.className = "fa-solid fa-check";
        btn.classList.add("is-copy-success");
        setTimeout(function () {
            if (icon) icon.className = "fa-regular fa-copy";
            btn.classList.remove("is-copy-success");
        }, 2000);
    };

    var fallbackCopy = function () {
        var ta = document.createElement("textarea");
        ta.value = text;
        ta.style.cssText = "position:fixed;top:0;left:0;opacity:0;";
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(ta);
        showSuccess();
    };

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(showSuccess).catch(fallbackCopy);
    } else {
        fallbackCopy();
    }
}
