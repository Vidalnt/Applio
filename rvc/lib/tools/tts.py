import sys
import asyncio
import edge_tts
import os


async def main():
    # Parse command line arguments
    tts_file = str(sys.argv[1])
    text = str(sys.argv[2])
    voice = str(sys.argv[3])
    rate = int(sys.argv[4])
    output_file = str(sys.argv[5])

    rates = f"+{rate}%" if rate >= 0 else f"{rate}%"
    if tts_file and os.path.exists(tts_file):
        text = ""
        try:
            with open(tts_file, "r", encoding="utf-8") as file:
                text = file.read()
        except UnicodeDecodeError:
            with open(tts_file, "r") as file:
                text = file.read()
    await edge_tts.Communicate(text, voice, rate=rates).save(output_file)
    # print(f"TTS with {voice} completed. Output TTS file: '{output_file}'")


if __name__ == "__main__":
    asyncio.run(main())
