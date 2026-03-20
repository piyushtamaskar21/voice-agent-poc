import pyaudio

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(
        i,
        info["name"],
        "in=", int(info.get("maxInputChannels", 0)),
        "out=", int(info.get("maxOutputChannels", 0)),
        "rate=", int(info.get("defaultSampleRate", 0)),
    )

print("Default output:", p.get_default_output_device_info())
