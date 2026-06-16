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

/**
 * Handle tabs for the "Across the sciences" section.
 */
document.addEventListener("DOMContentLoaded", function() {
    var tabs = document.querySelectorAll(".fury-sciences__tab");
    var displayImg = document.getElementById("fury-sciences-img");
    var displayTitle = document.getElementById("fury-sciences-title");
    var displayDesc = document.getElementById("fury-sciences-desc");

    if (!tabs.length || !displayImg) return;

    var contentData = {
        "engineering": {
            title: "Engineering",
            desc: "Robot-arm kinematics, assemblies and simulation output, animated as scene graphs."
        },
        "physics": {
            title: "Physics",
            desc: "Simulate particle collisions, wave propagation, and complex physical phenomena."
        },
        "chemistry": {
            title: "Chemistry",
            desc: "Visualize molecular structures, protein folding, and chemical reactions."
        },
        "astronomy": {
            title: "Astronomy",
            desc: "Render galaxies, orbital mechanics, and volumetric stellar data."
        },
        "aerospace": {
            title: "Aerospace",
            desc: "Visualize flow dynamics, wind tunnel simulations, and orbital trajectories."
        },
        "biology": {
            title: "Biology",
            desc: "Explore cellular structures, tissue imaging, and medical scans in 3D."
        },
        "data-science": {
            title: "Data science",
            desc: "Plot massive multi-dimensional datasets with GPU-accelerated rendering."
        },
        "network-science": {
            title: "Network science",
            desc: "Navigate complex interconnected graphs with millions of nodes and edges."
        },
        "mathematics": {
            title: "Mathematics",
            desc: "Render complex functions, manifolds, and topological structures."
        }
    };

    var currentIndex = 0;
    var intervalId = null;
    var delay = 2500; // 2.5 seconds delay
    var carouselContainer = document.querySelector(".fury-sciences__tabs");
    var userInteracted = false;

    // --- Preload images for seamless transitions ---
    var currentSrc = displayImg.getAttribute("src");
    var basePath = "";
    if (currentSrc) {
        var parts = currentSrc.split("popup/");
        if (parts.length === 2) {
            basePath = parts[0] + "popup/";
            Object.keys(contentData).forEach(function(key) {
                var img = new Image();
                img.src = basePath + key + ".gif";
            });
        }
    }

    function startCarousel() {
        if (intervalId) return;

        intervalId = setInterval(function() {
            currentIndex = (currentIndex + 1) % tabs.length;
            activateTab(tabs[currentIndex]);
        }, delay);
    }

    function stopCarousel() {
        if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
        }
    }

    function activateTab(tab) {
        tabs.forEach(function(t) {
            t.classList.remove("active");
            t.setAttribute("aria-selected", "false");
        });
        tab.classList.add("active");
        tab.setAttribute("aria-selected", "true");

        var target = tab.getAttribute("data-target");
        var data = contentData[target];

        if (data) {
            displayTitle.innerText = data.title;
            displayDesc.innerText = data.desc;

            if (basePath) {
                displayImg.setAttribute("src", basePath + target + ".gif");
            } else {
                // Fallback if structure was different
                var fallbackSrc = displayImg.getAttribute("src");
                var fallbackParts = fallbackSrc.split("popup/");
                if (fallbackParts.length === 2) {
                    displayImg.setAttribute("src", fallbackParts[0] + "popup/" + target + ".gif");
                }
            }
        }
    }

    tabs.forEach(function(tab, index) {
        tab.addEventListener("click", function() {
            currentIndex = index;
            activateTab(this);
            // stop auto-play permanently when user takes manual control
            userInteracted = true;
            stopCarousel();
        });
    });

    if (carouselContainer) {
        // Pause auto-play while hovered or focused
        var pauseCarousel = function() {
            if (!userInteracted) stopCarousel();
        };
        var resumeCarousel = function() {
            if (!userInteracted) startCarousel();
        };

        carouselContainer.addEventListener("mouseenter", pauseCarousel);
        carouselContainer.addEventListener("mouseleave", resumeCarousel);
        carouselContainer.addEventListener("focusin", pauseCarousel);
        carouselContainer.addEventListener("focusout", resumeCarousel);
    }

    startCarousel();
});

/**
 * Handle tabs for the installation commands section.
 */
document.addEventListener("DOMContentLoaded", function() {
    var terminalTabs = document.querySelectorAll(".fury-cta__tab");
    if (!terminalTabs.length) return;

    terminalTabs.forEach(function(tab) {
        tab.addEventListener("click", function() {
            var targetId = this.getAttribute("data-target");

            // Remove active class from all tabs and contents
            document.querySelectorAll(".fury-cta__tab").forEach(function(t) {
                t.classList.remove("active");
                t.setAttribute("aria-selected", "false");
            });
            document.querySelectorAll(".fury-cta__terminal").forEach(function(c) {
                c.classList.remove("active");
            });

            // Add active class to clicked tab and corresponding content
            this.classList.add("active");
            this.setAttribute("aria-selected", "true");
            var targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add("active");
            }
        });
    });
});
