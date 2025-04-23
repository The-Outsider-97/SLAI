import librosa  # For audio processing
import mido  # For MIDI processing
import music21  # For sheet music processing
import random
import re
import math
import json
import logging
import numpy as np

from collections import defaultdict

from models.slai_lm import SLAILM
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.knowledge_agent import KnowledgeAgent

class Musician:
    """
    A model designed to empower musicians in their growth and creative endeavors.
    """

    def __init__(self, shared_memory, agent_factory, expert_level=None, instrument=None):
        """
        Initializes the Musician model.
        """
        if expert_level is None or instrument is None:
            self.defer_setup = True
        else:
            self.defer_setup = False
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.slailm = SLAILM(shared_memory, agent_factory)
        self.reasoner = ReasoningAgent(shared_memory, agent_factory, tuple_key="music|property|value")
        self.knowledge_base = KnowledgeBase()  # Initialize the knowledge base
        self.output_module = OutputModule()
        self.music_theory_engine = MusicTheoryEngine() # music theory

        if expert_level is None:
            print("\n🎵 Welcome to the SLAI Musician Model 🎵")
            print("Please select your musical experience level:")
            print("1. Beginner\n2. Intermediate\n3. Advanced")
            level_choice = input("Enter choice (1-3): ").strip()
            self.expert_level = ["beginner", "intermediate", "advanced"][int(level_choice) - 1]
        else:
            self.expert_level = expert_level

        if instrument is None:
            print("\nWhat instrument do you play?")
            print("1. Piano\n2. Guitar\n3. Violin\n4. Cello\n5. Bass\n6. Drum\n7. Vocalist")
            instrument_choice = input("Enter choice (1-7): ").strip()
            self.instrument = ["piano", "guitar", "violin", "cello", "bass", "drum", "vocalist"][int(instrument_choice) - 1]
        else:
            self.instrument = instrument

    def interactive_setup(self):
        if self.defer_setup:
            # Replace this with a non-blocking UI dialog from Qt
            self.expert_level = "beginner"
            self.instrument = "piano"

    def process_input(self, input_data, input_type):
        """
        Processes the input data based on the specified input type.

        Args:
            input_data: The input data (audio, MIDI, text, or sheet music).
            input_type: The type of input data ('audio', 'midi', 'text', or 'sheet_music').

        Returns:
            The processed input data.
        """
        if input_type == 'audio':
            return self.process_audio(input_data)
        elif input_type == 'midi':
            return self.process_midi(input_data)
        elif input_type == 'text':
            return self.process_text(input_data)
        elif input_type == 'sheet_music':
            return self.process_sheet_music(input_data)
        else:
            raise ValueError(f"Invalid input type: {input_type}")

    def analyze_performance(self, input_data, input_type):
        """
        Analyzes a musical performance and provides feedback.

        Args:
            input_data:  The input performance data (audio or MIDI).
            input_type: The type of input data ('audio' or 'midi').

        Returns:
            A dictionary containing the analysis results and feedback.
        """
        if input_type not in ['audio', 'midi']:
            raise ValueError("Input type must be 'audio' or 'midi' for performance analysis.")

        processed_input = self.process_input(input_data, input_type)
        analysis_results = {}

        if input_type == 'audio':
            analysis_results = self.analyze_audio_performance(processed_input)
        elif input_type == 'midi':
            analysis_results = self.analyze_midi_performance(processed_input)

        feedback = self.generate_feedback(analysis_results, input_type)
        return {"analysis": analysis_results, "feedback": feedback}

    def generate_sheet_music(self, prompt):
        """
        Generates sheet music based on a text prompt.

        Args:
            prompt: A text prompt describing the desired music.

        Returns:
            The generated sheet music in a MusicXML format
        """
        from src.agents.perception.modules.transformer import Transformer
        from src.agents.qnn_agent import QNNAgent
        
        self.transformer = Transformer(num_layers=4)
        self.qrnn = QNNAgent(shared_memory, agent_factory, config={"num_qubits": 4})
        tokens = tokenize_prompt(prompt)
        sequence = self.transformer.forward(tokens)
        generated_notes = decode_music_tokens(sequence)
        return music21.stream.Score(generated_notes).write('musicxml')

    def process_audio(self, audio_data):
        """
        Processes audio data using Librosa.

        Args:
            audio_data: The audio data.

        Returns:
            The processed audio data.
        """
        # Example: Load audio file using librosa
        # y, sr = librosa.load(audio_file)
        # Placeholder: Replace with actual audio processing logic
        print("Processing audio data...")
        return audio_data

    def process_midi(self, midi_data):
        """
        Processes MIDI data using mido.

        Args:
            midi_data: The MIDI data.

        Returns:
            The processed MIDI data.
        """
        # Example: Process MIDI messages
        # for msg in midi_file.play():
        #     print(msg)
        print("Processing MIDI data...")
        return midi_data

    def process_text(self, text_data):
        """
        Processes text data.

        Args:
            text_data: The text data.

        Returns:
            The processed text data.
        """
        # Placeholder:  text processing
        print("Processing text data...")
        return self.slailm.process_input("musical_prompt", text_data)

    def process_sheet_music(self, sheet_music_data):
        """
        Processes sheet music data using music21.

        Args:
            sheet_music_data: The sheet music data.

        Returns:
            The processed sheet music data.
        """
        # Example: Parse a MusicXML file
        # score = music21.converter.parse(sheet_music_file)
        print("Processing sheet music data...")
        return sheet_music_data

    def analyze_audio_performance(self, audio_path):
        """
        Analyzes audio performance data.

        Args:
            audio_data: The processed audio data.

        Returns:
            A dictionary containing the analysis results.
        """
        y, sr = librosa.load(audio_path)
        pitches = librosa.yin(y, fmin=50, fmax=1000)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        return {
            "pitch": pitches.tolist(),
            "rhythm": beats.tolist(),
            "timbre": mfcc.mean(axis=1).tolist()
        }

    def analyze_midi_performance(self, midi_data):
        """
        Analyzes MIDI performance data.

        Args:
            midi_data: The processed MIDI data.

        Returns:
            A dictionary containing the analysis results.
        """
        notes = []
        for msg in midi_data:
            if msg.type == 'note_on':
                notes.append({
                    "note": msg.note,
                    "velocity": msg.velocity,
                    "time": msg.time
                })
        return {"notes": notes}

    def compare_with_reference(self, reference, performance):
        # Compute duration/velocity deltas
        ...
        return {"timing_accuracy": ..., "velocity_accuracy": ...}

    def generate_feedback(self, analysis_results, input_type):
        """
        Generates feedback based on the analysis results.

        Args:
            analysis_results: A dictionary containing the analysis results.
            input_type: The type of input data ('audio' or 'midi').

        Returns:
            A string containing the feedback.
        """
        prompt = f"Evaluate this {input_type} performance: {json.dumps(analysis_results)}"
        result = self.slailm.forward_pass({"raw_text": prompt})
        return result["text"]

    def get_practice_recommendations(self, skill_level, instrument, focus_area=None):
        """
        Gets practice recommendations based on the user's skill level, instrument, and focus area.

        Args:
            skill_level: The user's skill level (e.g., 'beginner', 'intermediate', 'advanced').
            instrument: The instrument the user plays (e.g., 'piano', 'guitar', 'violin').
            focus_area: The area the user wants to focus on (e.g., 'technique', 'improvisation', 'sight-reading').

        Returns:
            A list of practice recommendations.
        """
        # Placeholder:  Return practice recommendations from the knowledge base
        return self.knowledge_base.get_practice_recommendations(skill_level, instrument, focus_area)

    def explain_music_theory(self, concept):
        """
        Explains a music theory concept.

        Args:
            concept: The music theory concept to explain (e.g., 'harmony', 'melody', 'rhythm').

        Returns:
            A string containing the explanation.
        """
        # Placeholder: Return explanation from the knowledge base
        return self.knowledge_base.explain_music_theory(concept)

    def analyze_musical_style(self, piece):
        """
        Analyzes the musical style of a given piece.

        Args:
            piece: The musical piece (e.g., a score, an audio recording).

        Returns:
            A string describing the musical style.
        """
        # Placeholder: Analyze the musical style
        return "Analyzing musical style... (Place holder)"

    def set_goals_and_track_progress(self, goals):
        """
        Sets musical goals and tracks the user's progress.

        Args:
            goals: A list of goals.

        Returns:
            A dictionary containing the goals and progress.
        """
        facts = extract_music_facts(piece)  # e.g. [('piece', 'has_tempo', 'allegro')]
        self.reasoner.insert_knowledge(facts)
        inference = self.reasoner.query(("piece", "is_style", "?"))
        return str(inference)

class KnowledgeBase(KnowledgeAgent):
    """
    A class to represent the knowledge base.
    """

    def __init__(self):
        """
        Initializes the knowledge base.
        """
        self.music_theory_data = {
            "harmony": "Harmony is the study of chords and their relationships.",
            "melody": "Melody is a sequence of single notes that create a musical idea.",
            "rhythm": "Rhythm is the pattern of sounds and silences in time.",
            "counterpoint": "Counterpoint is the relationship between two or more musical lines (or voices) which are harmonically interdependent yet independent in rhythm and contour"
        }
        self.style_data = {
            "jazz": "Jazz is characterized by improvisation, syncopation, and swing.",
            "classical": "Classical music emphasizes formal structure, balance, and clarity.",
            "pop": "Pop music is typically song-based, with catchy melodies and simple harmonies.",
            "blues": "Blues is characterized by the use of blue notes, a 12-bar structure, and specific chord progressions."
        }
        self.practice_recommendations = {
            "beginner": {
                "piano": [
                    "Practice scales and chords daily.",
                    "Learn simple pieces to develop technique.",
                    "Use a metronome to improve timing."
                ],
                "guitar": [
                    "Learn basic chords (e.g., C, G, D, Em, Am).",
                    "Practice strumming patterns.",
                    "Play simple melodies."
                ],
                "violin": [
                    "Practice open strings and bowing technique.",
                    "Learn simple scales (e.g., G major, D major).",
                    "Work on intonation exercises."
                ]
            },
            "intermediate": {
                "piano": [
                    "Learn more complex chords and voicings.",
                    "Study counterpoint and harmony.",
                    "Practice sight-reading."
                ],
                "guitar": [
                    "Learn barre chords and advanced strumming techniques.",
                    "Explore different scales and modes.",
                    "Begin improvising over chord progressions."
                ],
                 "violin": [
                    "Practice advanced bowing techniques (e.g., détaché, martelé).",
                    "Learn more complex scales and arpeggios.",
                    "Work on vibrato and expressive playing."
                ]
            },
            "advanced": {
                "piano": [
                    "Master advanced repertoire.",
                    "Develop your own musical style.",
                    "Compose and arrange music."
                ],
                "guitar": [
                    "Develop advanced improvisation skills.",
                    "Explore complex harmonies and rhythms.",
                    "Compose original music in various styles."
                ],
                "violin": [
                    "Perform challenging concert repertoire.",
                    "Explore extended techniques and contemporary music.",
                    "Develop a unique artistic voice."
                ]
            }
        }

    def get_practice_recommendations(self, skill_level, instrument, focus_area=None):
        """
        Gets practice recommendations based on the user's skill level, instrument, and focus area.

        Args:
            skill_level: The user's skill level ('beginner', 'intermediate', 'advanced').
            instrument: The instrument the user plays.
            focus_area: The area the user wants to focus on.

        Returns:
            A list of practice recommendations.
        """
        if skill_level in self.practice_recommendations and instrument in self.practice_recommendations[skill_level]:
            recommendations = self.practice_recommendations[skill_level][instrument]
            if focus_area:
                #  filter recommendations based on focus_area.
                filtered_recommendations = [rec for rec in recommendations if focus_area.lower() in rec.lower()]
                return filtered_recommendations if filtered_recommendations else recommendations
            return recommendations
        return ["No practice recommendations found for the given skill level and instrument."]

    def explain_music_theory(self, concept):
        """
        Explains a music theory concept.

        Args:
            concept: The music theory concept to explain.

        Returns:
            A string containing the explanation.
        """
        return self.music_theory_data.get(concept.lower(), "Concept not found in knowledge base.")
    
    def get_musical_style(self, style):
        """
        Gets a description of a musical style

        Args:
            style: The style of music.

        Returns:
            A string containing the style description.
        """
        return self.style_data.get(style.lower(), "Style not found in knowledge base")

class OutputModule:
    """
    A class to handle output generation.
    """

    def generate_text_output(self, text):
        """
        Generates text output.

        Args:
            text: The text to output.
        """
        print(f"Text Output: {text}")

    def generate_sheet_music_output(self, musicxml_content):
        """
        Generates sheet music output.

        Args:
            musicxml_content: The MusicXML content.
        """
        # Placeholder:  Write MusicXML to a file or display it
        print("Generating sheet music output...")
        # print(musicxml_content) #  For debugging
        filename = "generated_music.xml"
        with open(filename, "w") as f:
            f.write(musicxml_content)
        print(f"Sheet music saved to {filename}")

    def generate_audio_output(self, audio_data):
        """
        Generates audio output.

        Args:
            audio_data: The audio data.
        """
        # Placeholder: Use a synthesizer to play audio
        print("Generating audio output...")

    def generate_visualization(self, data, visualization_type):
        """
        Generates a visualization.

        Args:
            data: The data to visualize.
            visualization_type: The type of visualization (e.g., 'spectrogram', 'waveform').
        """
        # Placeholder:  visualization using matplotlib or other libraries
        print(f"Generating {visualization_type} visualization...")

class MusicTheoryEngine:
    """
    A class to represent a music theory engine.
    """
    def __init__(self):
        # Initialize music theory data structures (e.g., chord dictionaries, scale lists)
        self.chords = {
            "major": [0, 4, 7],  # Major triad interval pattern
            "minor": [0, 3, 7],  # Minor triad interval pattern
            "dominant7": [0, 4, 7, 10],
            "major7": [0,4,7,11],
            "minor7": [0, 3, 7, 10],
        }
        self.scales = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],  # Natural minor
            "pentatonic_major": [0, 2, 4, 7, 9],
            "pentatonic_minor": [0, 3, 5, 7, 10],
        }

    def get_chord_notes(self, root_note, chord_type):
        """
        Calculates the notes in a chord.

        Args:
            root_note: The root note of the chord (e.g., 'C4', 'G3').
            chord_type: The type of chord (e.g., 'major', 'minor', 'dominant7').

        Returns:
            A list of notes in the chord (e.g., ['C4', 'E4', 'G4']).  Returns empty list if chord type is invalid.
        """
        if chord_type not in self.chords:
            print(f"Warning: Chord type '{chord_type}' not found.")
            return []

        root_pitch = music21.note.Note(root_note).pitch.midi  # Get MIDI pitch number
        intervals = self.chords[chord_type]
        notes = []
        for interval in intervals:
            pitch = root_pitch + interval
            notes.append(music21.midi.midiToNoteName(pitch))  # Convert MIDI pitch back to note name
        return notes

    def get_scale_notes(self, root_note, scale_type):
        """
        Calculates the notes in a scale.

        Args:
            root_note: The root note of the scale.
            scale_type: The type of scale (e.g., 'major', 'minor').

        Returns:
            A list of notes in the scale. Returns empty list if scale type is invalid.
        """
        if scale_type not in self.scales:
            print(f"Warning: Scale type '{scale_type}' not found.")
            return []
        root_pitch = music21.note.Note(root_note).pitch.midi
        intervals = self.scales[scale_type]
        notes = []
        for interval in intervals:
            pitch = root_pitch + interval
            notes.append(music21.midi.midiToNoteName(pitch))
        return notes

    def identify_harmony(notes):
        # Mapping of note names to semitone values (0-11) with enharmonic equivalents
        note_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11,
            'B#': 0, 'E#': 5, 'Fb': 4
        }
        
        # Reverse mapping for converting back to note names (prefer sharps)
        reverse_note_map = {
            0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
            6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
        }
        
        # Chord patterns with intervals relative to root (sorted)
        chord_patterns = [
            {'intervals': [4, 7], 'name': 'Major'},
            {'intervals': [3, 7], 'name': 'Minor'},
            {'intervals': [3, 6], 'name': 'Diminished'},
            {'intervals': [4, 8], 'name': 'Augmented'},
            {'intervals': [4, 7, 10], 'name': '7'},
            {'intervals': [4, 7, 11], 'name': 'Maj7'},
            {'intervals': [3, 7, 10], 'name': 'm7'},
            {'intervals': [3, 6, 9], 'name': 'dim7'},
            {'intervals': [2, 7], 'name': 'sus2'},
            {'intervals': [5, 7], 'name': 'sus4'},
            {'intervals': [7], 'name': '5'},
        ]
        
        # Convert input notes to numerical values
        values = []
        for note in notes:
            # Normalize accidentals and handle Unicode characters
            normalized = note.replace('♯', '#').replace('♭', 'b')
            if normalized in note_map:
                values.append(note_map[normalized])
            else:
                return "Unknown"
        
        if not values:
            return "Unknown"
        
        # Get unique, sorted note values (mod 12)
        unique_values = sorted(list({v % 12 for v in values}))
        
        # Check for chords
        for root in unique_values:
            # Calculate intervals from current root
            intervals = sorted([(v - root) % 12 for v in unique_values])
            # Remove root (0) and check remaining intervals
            chord_intervals = [i for i in intervals if i != 0]
            
            # Check against known chord patterns
            for pattern in chord_patterns:
                if chord_intervals == pattern['intervals']:
                    root_name = reverse_note_map[root]
                    return f"{root_name} {pattern['name']}"
        
        # Check for scales if no chord found
        if len(unique_values) == 7:
            # Major scale check
            for root in unique_values:
                major_scale = sorted([(root + offset) % 12 for offset in [0, 2, 4, 5, 7, 9, 11]])
                if unique_values == major_scale:
                    return f"{reverse_note_map[root]} Major Scale"
            
            # Natural minor scale check
            for root in unique_values:
                minor_scale = sorted([(root + offset) % 12 for offset in [0, 2, 3, 5, 7, 8, 10]])
                if unique_values == minor_scale:
                    return f"{reverse_note_map[root]} Natural Minor Scale"
        
        # Check for power chords (two notes with perfect fifth)
        if len(unique_values) == 2:
            interval = (unique_values[1] - unique_values[0]) % 12
            if interval == 7:
                return f"{reverse_note_map[unique_values[0]]} power chord"
            if interval == 5:  # Check inverted fifth (e.g., G below C)
                return f"{reverse_note_map[unique_values[1]]} power chord"
        
        # Check for single note
        if len(unique_values) == 1:
            return f"{reverse_note_map[unique_values[0]]} (Single Note)"
        
        return "Unknown"

    def generate_chord_progression(key='C Major', progression='I-IV-V-I'):
        """Generate diatonic chord progressions in any major/minor key.
        
        Args:
            key (str): Musical key (e.g., 'C Major', 'A Minor')
            progression (str): Roman numeral progression (e.g., 'I-V-vi-IV')
        
        Returns:
            list: Chord names in the progression
        """
        # Music theory configuration
        NOTE_MAP = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
            'B#': 0, 'Cb': 11, 'E#': 5, 'Fb': 4
        }
        
        REVERSE_NOTE_MAP = {
            0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
            6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
        }
        
        SCALE_INTERVALS = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10]
        }
        
        DIATONIC_CHORDS = {
            'major': ['Major', 'minor', 'minor', 'Major', 'Major', 'minor', 'dim'],
            'minor': ['minor', 'dim', 'Major', 'minor', 'minor', 'Major', 'Major']
        }
        
        ROMAN_MAP = {
            'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6,
            'i': 0, 'ii': 1, 'iii': 2, 'iv': 3, 'v': 4, 'vi': 5, 'vii': 6
        }

        # Parse input key
        try:
            root_name, scale_type = key.split()
            root_name = root_name.replace('♯', '#').replace('♭', 'b')
            scale_type = scale_type.lower()
            root_value = NOTE_MAP[root_name]
        except:
            return []

        # Generate scale notes
        scale_values = [(root_value + i) % 12 for i in SCALE_INTERVALS[scale_type]]
        scale_notes = [REVERSE_NOTE_MAP[v] for v in scale_values]

        # Create diatonic chord list
        chords = [
            f"{scale_notes[i]} {DIATONIC_CHORDS[scale_type][i]}" 
            for i in range(7)
        ]

        # Parse progression pattern
        progression_chords = []
        for numeral in progression.split('-'):
            # Clean and validate numeral
            clean_num = ''.join(c for c in numeral if c.isalpha()).upper()
            degree = ROMAN_MAP.get(clean_num, None)
            
            if degree is not None and 0 <= degree < 7:
                progression_chords.append(chords[degree])
        
        return progression_chords if progression_chords else []

    def generate_melody(key='C Major', tempo=120, time_signature='4/4', measures=4):
        """Generate a melody with musical structure and phrasing.
        
        Args:
            key (str): Musical key (e.g., 'C Major', 'A Minor')
            tempo (int): Beats per minute
            time_signature (str): Musical time signature
            measures (int): Number of measures
            
        Returns:
            list: List of (note, duration) tuples with rhythmic values
        """
        # Music theory configuration
        NOTE_MAP = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
            'B#': 0, 'Cb': 11, 'E#': 5, 'Fb': 4
        }
        
        REVERSE_NOTE_MAP = {
            0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
            6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
        }
        
        SCALE_INTERVALS = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10]
        }
        
        RHYTHM_PATTERNS = {
            '4/4': [
                ['whole'],
                ['half', 'half'],
                ['quarter']*4,
                ['eighth']*8,
                ['dotted-quarter', 'eighth', 'quarter', 'quarter'],
                ['quarter', 'dotted-quarter', 'eighth']
            ],
            '3/4': [
                ['dotted-half'],
                ['quarter']*3,
                ['half', 'quarter']
            ],
            '2/4': [
                ['half'],
                ['quarter']*2,
                ['eighth']*4,
                ['dotted-eighth', 'sixteenth', 'eight', 'eight']
            ]
        }
        
        BEAT_DURATIONS = {
            'whole': 4.0,
            'dotted-half': 3.0,
            'half': 2.0,
            'dotted-quarter': 1.5,
            'quarter': 1.0,
            'eighth': 0.5,
            'sixteenth': 0.25
        }

        # Parse input key
        try:
            root_name, scale_type = key.split()
            root_name = root_name.replace('♯', '#').replace('♭', 'b')
            scale_type = scale_type.lower()
            root_value = NOTE_MAP[root_name]
        except:
            return []

        # Generate scale notes
        scale_values = [(root_value + i) % 12 for i in SCALE_INTERVALS[scale_type]]
        scale_notes = [REVERSE_NOTE_MAP[v] for v in scale_values]
        tonic = scale_notes[0]

        # Determine time signature parameters
        beats_per_measure, beat_unit = map(int, time_signature.split('/'))
        total_beats = measures * beats_per_measure

        # Generate rhythmic pattern
        rhythm_pattern = random.choice(RHYTHM_PATTERNS[time_signature])
        rhythm = []
        while sum(BEAT_DURATIONS[d] for d in rhythm) < total_beats:
            rhythm.extend(random.choice(RHYTHM_PATTERNS[time_signature]))
        rhythm = rhythm[:sum(BEAT_DURATIONS[d] for d in rhythm) <= total_beats]

        # Create melodic contour (ascending -> descending)
        contour = []
        current_octave = 4
        direction = 1  # Start ascending
        
        for i, duration in enumerate(rhythm):
            # Change direction at midpoint
            if i == len(rhythm)//2:
                direction = -1
                
            # Select note from scale with some variations
            note_choices = scale_notes.copy()
            
            # Add occasional passing tones
            if random.random() < 0.2:
                note_choices += [REVERSE_NOTE_MAP[(root_value + random.choice([-1,1])) % 12]]
                
            # Add octave variations
            if random.random() < 0.3:
                current_octave += random.choice([-1, 1])
                current_octave = max(3, min(5, current_octave))
                
            # Select note based on contour direction
            base_note = scale_notes[min(len(scale_notes)-1, max(0, i % len(scale_notes)))]
            note = random.choice([
                base_note,
                scale_notes[(scale_notes.index(base_note) + direction) % len(scale_notes)],
                tonic  # Emphasize tonic
            ])
            
            contour.append(f"{note}{current_octave}")

        # Add phrase structure (question-answer)
        melody = []
        phrase_length = len(contour)//2
        for i, (note, duration) in enumerate(zip(contour, rhythm)):
            # End first phrase with leading tone
            if i == phrase_length - 1:
                note = scale_notes[-1] + str(current_octave)
            
            # End melody with tonic
            if i == len(contour) - 1:
                note = tonic + str(current_octave)
            
            # Add occasional rests
            if random.random() < 0.1:
                melody.append(('R', duration))
            else:
                melody.append((note, duration))

        return melody


if __name__ == "__main__":
    Musician = Musician(shared_memory=None, agent_factory=None)
