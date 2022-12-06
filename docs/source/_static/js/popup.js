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

function calculatePositionForPopup(elBounds, popupWidth = 350, popupHeight = 300) {

    let popupBounds = {
        height: popupHeight,
        width: popupWidth
    }

    console.log(elBounds.top)

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
function removePopup() {
    $('#tutorial-popup').addClass('tutorial-popup-inactive');
    $('#tutorial-popup').removeClass('tutorial-popup-active');
}

// Open the popup on cell-layout hover
$('img.cell-layout').hover(function () {
        // over,
        showPopup(this);
    }, () => {}
);

// Close the popup when exited from the popup
$('#tutorial-popup').hover(() => {}, function () {
        // out
        removePopup()
    }
);

// To remove the popup when clicked outside
$(document).click(function (e) { 
    e.preventDefault();
    if($(e.target).closest('#tutorial-popup').length == 0) {
        removePopup()
    }
});

