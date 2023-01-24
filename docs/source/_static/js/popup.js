const POPUP_TYPE = {
    TUTORIAL: 'tutorial'
}

const ACTIVE_POPUP = {
    TUTORIAL: false
}

const TUTORIALS = {
    // the keys are id of the element and the file and title will update accordingly
    engineering: {
        fileName: 'engineering',
        title: 'engineering',
        link: ''
    },
    physics: {
        fileName: 'physics',
        title: 'physics',
        link: ''
    },
    chemistry: {
        fileName: 'chemistry',
        title: 'chemistry',
        link: ''
    },
    astronomy: {
        fileName: 'astronomy',
        title: 'astronomy',
        link: ''
    },
    aerospace: {
        fileName: 'aerospace',
        title: 'aerospace',
        link: ''
    },
    biology: {
        fileName: 'biology',
        title: 'biology',
        link: ''
    },
    'data-science': {
        fileName: 'data-science',
        title: 'data-science',
        link: ''
    },
    'network-science': {
        fileName: 'network-science',
        title: 'network-science',
        link: ''
    },
    mathematics: {
        fileName: 'mathematics',
        title: 'mathematics',
        link: ''
    }
}

// checking is mobile
var isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)

function createTutorialPopup(tutorial, fromTop) {
    return `<div id="tutorial-popup" class="tutorial-popup ${fromTop ? 'tutorial-popup-top': 'tutorial-popup-bottom'}">
        <div class="gallery-box-title">${tutorial.title}</div>
        <img src="_static/images/sc-dom/popup/${tutorial.fileName}.gif" alt="">
    </div>`;
}

// open a popup
function showPopup(el, popupType = POPUP_TYPE.TUTORIAL) {

    let popupBounds = calculatePositionForPopup(el.getBoundingClientRect());

    // Injecting the element
    if (popupType === POPUP_TYPE.TUTORIAL) {

        // Checking for already active popup to make sure only one popup present
        if (ACTIVE_POPUP.TUTORIAL) {
            removePopup();
        }

        $('.scientific-domains').append(createTutorialPopup(TUTORIALS[el.id], popupBounds.fromTop));
        ACTIVE_POPUP.TUTORIAL = true;

        // Adding position
        $('#tutorial-popup').css({
            'top': `${popupBounds.top}px`,
            'left': `${popupBounds.left}px`,
            'animation-name': 'expand'
        });

        // Close the popup when exited from the popup
        $('#tutorial-popup').hover(() => {}, function () {
            removePopup();
        });

    }
}

function calculatePositionForPopup(elBounds, popupWidth = 350, popupHeight = 300) {

    let popupBounds = {
        height: popupHeight,
        width: popupWidth,
        fromTop: true
    }

    if (elBounds.top < 308) {
        // Open the popup on bottom
        popupBounds.top = elBounds.bottom + 8;
        popupBounds.bottom = elBounds.bottom + popupHeight + 8;
        popupBounds.fromTop = false;
    } else {
        popupBounds.top = elBounds.top - popupHeight - 8;
        popupBounds.bottom = elBounds.top - 8;
    }

    popupBounds.left = elBounds.left;
    popupBounds.right = elBounds.left + popupWidth;

    return popupBounds;
}

// remove the popup
function removePopup(popupType = POPUP_TYPE.TUTORIAL) {
    $(`#${popupType}-popup`).remove();
    ACTIVE_POPUP.TUTORIAL = false;
}

// Open the popup on cell-layout hover
$('img.cell-layout').hover(function () {
        // over,
        if (!isMobile) {
            showPopup(this);
        }
    }, () => {}
);

// Close the popup when exited from the popup
$('#tutorial-popup').hover(() => {}, function () {
        removePopup();
    }
);

// To remove the popup when clicked outside
$(document).click(function (e) {
    // We can append more of these to close the popups
    if($(e.target).closest('#tutorial-popup').length == 0) {
        removePopup();
    }
});

// To remove the popup when scrolled
$(document).scroll(function (e) { 
    if($(e.target).closest('#tutorial-popup').length == 0) {
        removePopup();
    }
});


