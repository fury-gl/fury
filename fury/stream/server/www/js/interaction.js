import {
  weelInterval,
  mouseInterval,
  socketAddr,
} from "/js/constants.js";
const clientSocket = new WebSocket(
  socketAddr);

export const  addInteraction = (videoEl)=>{
  let mouseLeftReleased = true;
  let mouseOutVideo = false;
  let mouseX = 0;
  let mouseY = 0;
  let ctrlKey = 0;
  let shiftKey = 0;
  let enableCallWheel = true;
  let enableCallMove = true;
  let currentWheelEventTotalDeltaY = 0;

  videoEl.addEventListener("wheel", (event) => {
    event.preventDefault();
    currentWheelEventTotalDeltaY += event.deltaY;
    if (!enableCallWheel) return;

    const data = {
      type: "weel",
      deltaY: currentWheelEventTotalDeltaY,
      timestampInMs: Date.now(),
    };
    const dataJson = JSON.stringify(data);
    currentWheelEventTotalDeltaY = 0;
    clientSocket.send(dataJson);
    enableCallWheel = false;
    setTimeout(() => (enableCallWheel = true), weelInterval);
  });

  videoEl.addEventListener("mousemove", (event) => {
    if (!enableCallMove) return;
    mouseOutVideo = false;
    ctrlKey = event.ctrlKey ? 1 : 0;
    shiftKey = event.shiftKey ? 1 : 0;
    const width = videoEl.offsetWidth;
    const height = videoEl.offsetHeight;
    mouseX = event.offsetX / width;
    mouseY = event.offsetY / height;
    mouseMoveCallback();
    enableCallMove = false;
    setTimeout(() => (enableCallMove = true), mouseInterval);
  });

  videoEl.addEventListener("mouseleave", (e) => (mouseOutVideo = false));

  const mouseMoveCallback = () => {
    if (mouseLeftReleased || mouseOutVideo) return;
    const data = {
      type: "mouseMove",
      x: mouseX,
      y: mouseY,
      ctrlKey: ctrlKey,
      shiftKey: shiftKey,
      timestampInMs: Date.now(),
    };
    const dataJson = JSON.stringify(data);
    clientSocket.send(dataJson);
  };

  videoEl.addEventListener("mousedown", (e) => {
    e.preventDefault();
    mouseLeftReleased = false;
    ctrlKey = e.ctrlKey ? 1 : 0;
    shiftKey = e.shiftKey ? 1 : 0;
    const mouseButton = e.button;
    const data = {
      type: "mouseLeftClick",
      on: 1,
      mouseButton: mouseButton,
      x: mouseX,
      y: mouseY,
      ctrlKey: ctrlKey,
      shiftKey: shiftKey,
      timestampInMs: Date.now(),
    };
    const dataJson = JSON.stringify(data);

    clientSocket.send(dataJson);
  });

  window.addEventListener("mouseup", (e) => {
    mouseLeftReleased = true;
    ctrlKey = e.ctrlKey ? 1 : 0;
    shiftKey = e.shiftKey ? 1 : 0;
    const mouseButton = e.button;
    const data = {
      type: "mouseLeftClick",
      on: 0,
      x: mouseX,
      y: mouseY,
      mouseButton: mouseButton,
      ctrlKey: ctrlKey,
      shiftKey: shiftKey,
      timestampInMs: Date.now(),
    };
    const dataJson = JSON.stringify(data);

    clientSocket.send(dataJson);
  });
}
