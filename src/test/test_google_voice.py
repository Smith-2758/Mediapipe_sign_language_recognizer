from gtts import gTTS
import os
import pygame

pygame.mixer.init()

def play_tts(text, filename="test_tts.mp3"):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove(filename)
        return True
    except Exception as e:
        print(f"gTTS Error: {e}")
        return False

if __name__ == '__main__':
    if play_tts("Hello, this is a test."):
        print("gTTS test successful.")
    else:
        print("gTTS test failed.")