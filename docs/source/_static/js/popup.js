const POPUP_TYPE = {
    TUTORIAL: 'tutorial'
}

const ACTIVE_POPUP = {
    TUTORIAL: false
}

function createTutorialPopup(fileName, title, link) {
    return `<div id="tutorial-popup" class="tutorial-popup">
        <div class="gallery-box-title">${title}</div>
        <img src="_static/images/${fileName}.gif" alt="">
    </div>`
}

// open a popup
function showPopup(el, popupType = POPUP_TYPE.TUTORIAL) {
    

    // Injecting the element
    if (popupType === POPUP_TYPE.TUTORIAL) {
        
        // Checking for already active popup to make sure only one popup present
        if (ACTIVE_POPUP.TUTORIAL) {
            removePopup()
        }
        
        $('.scientific-domains').append(createTutorialPopup('horse', 'engineering'));
        ACTIVE_POPUP.TUTORIAL = true
    }
    
    popupBounds = calculatePositionForPopup(el.getBoundingClientRect())

    // Adding position
    $('#tutorial-popup').css({
        'top': `${popupBounds.top}px`,
        'left': `${popupBounds.left}px`,
        'animation-name': 'expand'
    });


}

function calculatePositionForPopup(elBounds, popupWidth = 350, popupHeight = 300) {

    let popupBounds = {
        height: popupHeight,
        width: popupWidth
    }

    if (elBounds.top < 308) {
        // Open the popup on bottom
    } else {
        popupBounds.top = elBounds.top - popupHeight - 8
        popupBounds.bottom = elBounds.top - 8
    }

    popupBounds.left = elBounds.left
    popupBounds.right = elBounds.left + popupWidth


    return popupBounds
}

// remove the popup
function removePopup(popupType = POPUP_TYPE.TUTORIAL) {
    $(`#${popupType}-popup`).remove();
    ACTIVE_POPUP.TUTORIAL = false
}

// Open the popup on cell-layout hover
$('img.cell-layout').hover(function () {
        // over,
        showPopup(this);
    }, () => {}
);

// Close the popup when exited from the popup
$('#tutorial-popup').hover(() => {}, function () {
        removePopup()
    }
);

// To remove the popup when clicked outside
$(document).click(function (e) { 
    e.preventDefault();
    // We can append more of these to close the popups
    if($(e.target).closest('#tutorial-popup').length == 0) {
        removePopup()
    }
});

