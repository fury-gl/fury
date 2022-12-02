// open a popup on the given element of size 200x200
function showPopup(el) {
    $('#tutorial-popup').addClass('tutorial-popup-active');
    $('#tutorial-popup').removeClass('tutorial-popup-inactive');
    
    popupBounds = calculatePositionForPopup(el.getBoundingClientRect())

    // Adding position
    $('#tutorial-popup').css({
        'top': `${popupBounds.top}px`,
        'left': `${popupBounds.left}px`
    });
}

function calculatePositionForPopup(elBounds, popupWidth = 200, popupHeight = 200) {

    let popupBounds = {
        height: popupHeight,
        width: popupWidth
    }

    if (elBounds.top < 250) {
        // Open the popup on bottom
    } else {
        popupBounds.top = elBounds.top - popupHeight
        popupBounds.bottom = elBounds.top
    }

    if (elBounds.right < 250) {
        // Open the popup on left
    } else {
        popupBounds.left = elBounds.left
        popupBounds.right = elBounds.left + popupWidth
    }

    return popupBounds
}

// remove the popup
function removePopup(el) {

    

    $('#tutorial-popup').addClass('tutorial-popup-inactive');
    $('#tutorial-popup').removeClass('tutorial-popup-active');
}

// Open the popup on cell-layout hover
$('img.cell-layout').hover(function () {
        // over
        showPopup(this);
    }, function (evt) {
        // out
        
        removePopup(this);
    }
);

