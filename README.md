# Baller

A browser-based hand-tracking ball demo built with p5.js and MediaPipe Hands.

## Run

1. Start a local server:

   ```bash
   npm run dev
   ```

2. Open `http://localhost:4173`.
3. Allow webcam access when the browser prompts for it.
4. Use either hand to pinch near the ball and grab it, or bring an open palm under it to catch and support it.

## Notes

- The webcam feed is mirrored to feel more natural.
- Up to two hands are tracked at once, so you can pass the ball between hands.
- A pinch grabs the ball, while an open palm can catch it and let it rest on the hand surface.
- Hand motion is smoothed before it drives the throw velocity.
- The ball uses gravity, hand collision/support logic, wall collision, and a floor with damping so it settles when untouched.
