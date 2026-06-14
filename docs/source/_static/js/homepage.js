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

    tabs.forEach(function(tab) {
        tab.addEventListener("click", function() {
            tabs.forEach(function(t) { t.classList.remove("active"); });
            this.classList.add("active");

            var target = this.getAttribute("data-target");
            var data = contentData[target];
            
            if (data) {
                displayTitle.innerText = data.title;
                displayDesc.innerText = data.desc;
                
                var currentSrc = displayImg.getAttribute("src");
                var parts = currentSrc.split("popup/");
                if (parts.length === 2) {
                    displayImg.setAttribute("src", parts[0] + "popup/" + target + ".gif");
                }
            }
        });
    });
});
