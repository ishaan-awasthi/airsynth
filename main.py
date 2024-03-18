from funcs import *

def main():
    ZONE_FREQUENCIES = {
        "red" : [523.25, 659.25, 783.99],       # I - C (MAJOR)
        "orange" : [493.88, 587.33, 698.46],    # viiº - B (DIMINISHED)
        "yellow" : [440.00, 523.25, 659.25],    # vi - A (MINOR)
        "green" : [392.00, 493.88, 587.33],     # V - G (MAJOR)
        "blue" : [349.23, 440.00, 523.25],      # IV - F (MAJOR)
        "indigo" : [329.63, 392.00, 493.88],    # iii - E (MINOR)
        "purple" : [293.66, 349.23, 440.00],    # ii - D (MINOR)
        "pink" : [261.63, 329.63, 392.00],      # I - C (MAJOR)
    }
    prev_zone = current_zone = None
    show_colors = False

    cap = cv2.VideoCapture(0)
    LM = Landmarker()
    #ST = threading.Thread(target=melody, args=(ZONE_FREQUENCIES, current_zone))
    #ST.start()
    MelodyThread = MelodyStream(0, 1)
    #MelodyThread.start()
    ChordThread = WavePlayerLoop(0, 1, 1)
    #ChordThread.start()
    handedness = None

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        og_frame_data = frame
        h, w = frame.shape[:2]
        if show_colors:render_color(frame, h, w)
        LM.detect_async(og_frame_data)
        result = LM.result
        frame = render_landmarks(frame, result) 
        frame, current_zone = fingertip_y(frame, result, current_zone, h)

        try:
            handedness = getattr(result, 'handedness', None)[0][0].category_name
        except:
            pass

        
        if show_colors and handedness=="Left":
            if(current_zone != prev_zone):
                if MelodyThread.is_alive(): MelodyThread.join()
                MelodyThread = MelodyStream(ZONE_FREQUENCIES[current_zone][0], 0.5)
                MelodyThread.start()
            prev_zone = current_zone

        if show_colors and handedness=="Right":
            if(current_zone != prev_zone):
                if ChordThread.is_alive(): ChordThread.join()
                for tone in ZONE_FREQUENCIES[current_zone]:
                    ChordThread = WavePlayerLoop(tone, 1, 1.5)
                    ChordThread.start()
            prev_zone = current_zone
        """
        """

    # Display the frame with modifications for both hands
        cv2.imshow('Output', frame)
        key = cv2.waitKey(1)
        if key == ord('c'):
            show_colors = not show_colors
        if key == ord('q'):
            break
    cap.release()
    MelodyThread.join()
    cv2.destroyAllWindows

if __name__ == '__main__':
    main()



"""
if(current_zone != prev_zone):
                    ST.join()
                    ST = threading.Thread(target=melody, args=(ZONE_FREQUENCIES, current_zone))
                    ST.start()
                prev_zone = current_zone

"""




"""
        for hand_landmarks in result.hand_landmarks:
        # Determine if the detected hand is the right or left hand
            if result.handedness[0][0].display_name == 'Right':
                print("Processing Right hand")
                # Your logic for processing the Right hand goes here
            elif result.handedness[0][0].display_name == 'Left':
                print("Processing Left hand")
                # Your logic for processing the Left hand goes here

            # Perform fingertip_y function for each hand separately
            frame, current_zone = fingertip_y(frame.copy(), hand_landmarks, current_zone)

            # Merge the results of processing each hand onto the same frame
            if show_colors:
                render_color(frame)
                if(current_zone != prev_zone):
                    ST.join()
                    ST = threading.Thread(target=melody, args=(ZONE_FREQUENCIES, current_zone))
                    ST.start()
                prev_zone = current_zone

"""