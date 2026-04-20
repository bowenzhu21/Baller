# Baller

A browser-based hand-tracking ball demo built with p5.js and MediaPipe Hands.

## Run

1. Start a local server:

   ```bash
   npm run dev
   ```

2. Open `http://localhost:4173`.
3. Click `Start Camera` and allow webcam access.
4. Pinch near the ball to grab it, then release to throw it.

## Notes

- The webcam feed is mirrored to feel more natural.
- Hand motion is smoothed before it drives the throw velocity.
- The ball uses simple gravity, wall collision, and a floor with damping so it settles when untouched.
# Baller
