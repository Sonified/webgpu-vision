# Pre-Flight Checklist: WebGPU Vision Public Launch

## 1. Tune 1-Euro Filters
- [x] Allow min cutoff floor up to 5 for hands
- [x] Dial in params for smooth slow movements + responsive fast gestures

## 2. Fix Initialization Bug
- [x] Diagnose and fix the intermittent failure to initialize on page load
- [x] Root cause: dedupe bail in animate() returned before scheduling next rAF, killing the render loop
- [ ] Verify with 20+ cold loads

## 3. Investigate: Palm Detection Back on GPU
- [ ] Palm detection still does a GPU->CPU readback (mapAsync) to feed ORT
- [ ] Was taken off GPU previously -- may not have been working due to the init bug, not a GPU issue
- [ ] If the init bug fix resolves it, move palm detection back to full GPU (eliminates last readback)
- [ ] Would also reduce fan/thermal load

## 4. Z-Depth Beta: Launch Decision
- [ ] Decide: ship z-depth as a beta feature at launch or keep internal?

## 5. Hamburger Menu: Hand-Viz Demo
- [ ] Add hand detection confidence threshold slider
- [ ] Chat with AI about additional params worth exposing
- [ ] Localhost-only mode option (visible on localhost, hidden in production)
- [ ] Section dividers, vertical spacing polish

## 6. Hamburger Menu: Ball-Toss Demo
- [ ] Move current top-bar controls into hamburger
- [ ] Section dividers, match hand-viz styling

## 7. Lightweight Embed API
- [ ] Design the public API surface
- [ ] Consider what belongs in the library vs the demos

## 8. Consider: Gesture Recording Demo
- [ ] Could a stripped-down version of SpellARia's motion recorder work as a showcase demo?

## 9. Index Page
- [ ] Decide what goes on root index.html

## 10. Benchmarks
- [ ] Run latest numbers across model variants
- [ ] Compare against MediaPipe

## 11. First Post / Announcement
- [ ] Write post draft
- [ ] Record GIF animations
- [ ] Include links to live demos

## 12. File Organization + README
- [ ] Generate proper README.md
- [ ] Audit file tree for anything that shouldn't be public
- [ ] .claude/ directory MUST be gitignored

## 13. Repo Audit: Archive or Clean?
- [ ] Check repo size and large files in history
- [ ] Decision: clean this repo or start fresh?

## 14. 60fps Camera & Frame Latency Investigation
- [ ] Optimize for iPhone front-facing camera (60fps capable)
- [ ] Investigate MacBook Pro camera -- 30fps hardware or software limit?
- [ ] Research browser frame buffering / presentation latency

## 15. Go Public
- [ ] Verify models load from CDN, not repo
- [ ] Test ALL demo pages
- [ ] Fix wave pool canvas to resize with window
- [ ] Test in another browser + private/incognito window
- [ ] Toggle repo public
- [ ] Post on LinkedIn

---

# Post-Launch Roadmap

## TouchDesigner / Creative Tool Integration
- [ ] Build OSC output adapter for TouchDesigner, Max/MSP, etc.
- [ ] Visual artist with ~200K Instagram followers ready to build with it

## Knowledge Distillation: Custom Hand Model
- [ ] See WORK-PLAN.md "Future: knowledge distillation" section
