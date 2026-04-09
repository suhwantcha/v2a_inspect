import inspect
from elevenlabs.client import ElevenLabs

print("Music compose signature:", inspect.signature(ElevenLabs().music.compose))
print("SFX convert signature:", inspect.signature(ElevenLabs().text_to_sound_effects.convert))
