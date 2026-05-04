/**
 * Types text into an element character by character.
 * @param {HTMLElement} element - The DOM element to type into.
 * @param {string} text - The text to type.
 * @param {number} speed - The typing speed in ms per character.
 * @param {Function} callback - Optional callback when typing is complete.
 * @returns {Function} - A function to cancel the current typing animation.
 */
export function typeWriter(element, text, speed = 30, callback = null) {
  element.textContent = '';
  let i = 0;
  let cancelled = false;

  function type() {
    if (cancelled) return;
    
    if (i < text.length) {
      element.textContent += text.charAt(i);
      i++;
      setTimeout(type, speed);
    } else if (callback) {
      callback();
    }
  }

  type();

  return () => {
    cancelled = true;
  };
}