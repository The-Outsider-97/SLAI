from models.music.music_theory_engine import MusicTheoryEngine
import random
import os

engine = MusicTheoryEngine()

GENRES = [
    'blues', 'jazz', 'pop', 'funk',
    'classical', 'rock', 'hiphop', 'country',
    'techno', 'reggae', 'metal', 'r&b'
]
NUM_SAMPLES_PER_GENRE = 100_000
OUTPUT_DIR = "synthetic_dataset/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for genre in GENRES:
    genre_dir = os.path.join(OUTPUT_DIR, genre)
    os.makedirs(genre_dir, exist_ok=True)

    for i in range(NUM_SAMPLES_PER_GENRE):
        # 1. Random key
        key = random.choice(['C', 'D', 'E', 'F', 'G', 'A', 'B']) + random.choice(['', 'b', '#']) + ' major'

        # 2. Genre-specific progression
        if genre == 'blues':
            progression = engine.generate_chord_progression(key, progression_input='I-IV-I-I-IV-IV-I-I-V-IV-I-V')
        elif genre == 'jazz':
            progression = engine.generate_chord_progression(key, progression_input='ii-V-I-vi-ii-V-I')
        elif genre == 'pop':
            progression = engine.generate_chord_progression(key, progression_input='I-V-vi-IV')
        elif genre == 'funk':
            progression = engine.generate_chord_progression(key, progression_input='i7-vi7-ii7-V7')  # Example funk progression
        elif genre == 'classical':
            progression = engine.generate_chord_progression(key, progression_input='I-IV-V-I-vi-ii-V-I')  # Common classical progression
        elif genre == 'rock':
            progression = engine.generate_chord_progression(key, progression_input='I-V-vi-iii-IV-I-ii-V')  # Popular rock progression
        elif genre == 'hiphop':
            progression = engine.generate_chord_progression(key, progression_input='i-VII-VI-III')  # Common hip hop progression
        elif genre == 'country':
            progression = engine.generate_chord_progression(key, progression_input='I-IV-I-V')  # Classic country progression
        elif genre == 'techno':
            progression = engine.generate_chord_progression(key, progression_input='i-ii-IV-V')  # Example techno progression (often repetitive)
        elif genre == 'reggae':
            progression = engine.generate_chord_progression(key, progression_input='i-iv-I-V')  # Typical reggae progression
        elif genre == 'metal':
            progression = engine.generate_chord_progression(key, progression_input='i-VI-III-VII')  # Example metal progression (often power chords)
        elif genre == 'r&b':
            progression = engine.generate_chord_progression(key, progression_input='ii-V-I-IV')  # Common R&B progression
        else:
            progression = engine.generate_chord_progression(key, num_chords=8)

        # 3. Genre-specific melody
        if genre == 'jazz':
            melody = engine.generate_melody(key=key, measures=8, rhythm_complexity=0.7, contour_shape='random',
                                        range_octaves=2)  # Increased range for jazz
        elif genre == 'blues':
            melody = engine.generate_melody(key=key, measures=8, rhythm_complexity=0.5, contour_shape='flat',
                                        use_blue_notes=True)  # Blues often uses blue notes
        elif genre == 'pop':
            melody = engine.generate_melody(key=key, measures=8, rhythm_complexity=0.4, contour_shape='arch',
                                        range_octaves=1)  # Pop melodies are often catchy and simple
        elif genre == 'funk':
            melody = engine.generate_melody(key=key, measures=8, rhythm_complexity=0.8, contour_shape='random',
                                        range_octaves=2)  # Complex rhythms and wider range
        elif genre == 'classical':
            melody = engine.generate_melody(key=key, measures=16, rhythm_complexity=0.3, contour_shape='arch',
                                        range_octaves=2)  # Longer phrases and arched contours
        elif genre == 'rock':
            melody = engine.generate_melody(key=key, measures=8, rhythm_complexity=0.6, contour_shape='random',
                                        range_octaves=1)  # Energetic and driving
        elif genre == 'hiphop':
            melody = engine.generate_melody(key=key, measures=4, rhythm_complexity=0.7, contour_shape='stepwise',
                                        use_pentatonic=True)  # Often uses pentatonic scales
        elif genre == 'country':
            melody = engine.generate_melody(key=key, measures=8, rhythm_complexity=0.4, contour_shape='arch',
                                        range_octaves=1)  # Simple and lyrical
        elif genre == 'techno':
            melody = engine.generate_melody(key=key, measures=4, rhythm_complexity=0.9, contour_shape='stepwise',
                                        range_octaves=1)  # Repetitive and rhythmic
        elif genre == 'reggae':
            melody = engine.generate_melody(key=key, measures=8, rhythm_complexity=0.6, contour_shape='undulating',
                                        use_offbeat=True)  # Offbeat rhythms are key
        elif genre == 'metal':
            melody = engine.generate_melody(key=key, measures=8, rhythm_complexity=0.7, contour_shape='random',
                                        range_octaves=2, use_chromatic=True)  # Chromaticism and intensity
        elif genre == 'r&b':
            melody = engine.generate_melody(key=key, measures=8, rhythm_complexity=0.5, contour_shape='arch',
                                        range_octaves=1)  # Smooth and soulful
        else:
            melody = engine.generate_melody(key=key, measures=8, rhythm_complexity=0.3, contour_shape='random')

        # 4. Save melody+chords
        stream = melody
        for chord in progression:
            stream.append(chord)

        filepath = os.path.join(genre_dir, f"{genre}_{i}.mid")
        stream.write('midi', fp=filepath)
