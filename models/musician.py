import librosa  # For audio processing
import music21  # For sheet music processing
import mido
import random
import re
import time
import math
import json
import logging
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict, deque
from PyQt5.QtCore import QObject, QPropertyAnimation, QSize, Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QLabel, QWidget, QApplication, QVBoxLayout, QHBoxLayout

from models.slai_lm import SLAILM
from models.music.output_module import OutputModule
from models.music.music_theory_engine import MusicTheoryEngine
from models.music.audio_utils import AudioUtils
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.perception.modules.transformer import Transformer
from src.agents.qnn_agent import QNNAgent
from logs.logger import get_logger

logger = get_logger(__name__)

class KnowledgeBase(KnowledgeAgent): # Defined within musician.py in the provided code
    """ Specialized KnowledgeAgent for music-related information. """
    def __init__(self, shared_memory, agent_factory):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory)
        # Define music ontology relationships
        self.ontology = self._load_ontology()

        self.music_theory_data = self._load_music_theory_data() # Music theory concepts dictionary
        self.style_data = self._load_style_data() # Musical style descriptions

        # Load practice recommendations, potentially adding documents to the knowledge base
        self.practice_recommendations = self._load_practice_recommendations()

    def _load_ontology(self):
        try:
            with open('models/music/data/music_ontology.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Ontology file missing! Using minimal fallback.")
            return {"nodes": {}, "relationships": []}
    
    def get_related_concepts(self, concept):
        """Find all nodes and relationships connected to a concept."""
        related = []
        for rel in self.ontology["relationships"]:
            if rel["from"] == concept or rel["to"] == concept:
                related.append(rel)
        return related

    def _load_music_theory_data(self):
        try:
            with open('models/music/data/music_theory_data.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Music theory data not found. Using defaults.")
            return {
                #default fallback dict
                }

    def _load_style_data(self):
        try:
            with open('models/music/data/styles_data.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Musical styles data not found. Using defaults.")
            return {
                # default fallback dict
            }

    def _load_practice_recommendations(self):
        """Load practice recommendations from JSON file with fallback"""
        try:
            with open('models/music/data/practice_recommendations.json', 'r', encoding='utf-8') as f: # Path adjusted, assuming it's in a 'data' subdir
                data = json.load(f)
                recommendations = data.get('practice_recommendations', {})
                # Loaded recommendations as documents to the knowledge base
                for level, instruments in recommendations.items():
                    for instr, rec_list in instruments.items():
                        for rec in rec_list:
                            self.add_document(f"{level} {instr} practice: {rec}",
                                              metadata={'type': 'practice_tip', 'level': level, 'instrument': instr})
                return recommendations
        except FileNotFoundError:
            logger.warning("Practice recommendations file not found, using default data")
            return self._default_practice_recommendations()
        except json.JSONDecodeError as e:
            logger.error(f"Error loading practice recommendations: {e}")
            return self._default_practice_recommendations()

    def _default_practice_recommendations(self):
        """Fallback recommendations if JSON loading fails"""
        # Default recommendations as documents
        defaults = {
            "beginner": {
                "piano": ["Practice scales and chords daily"],
                "guitar": ["Learn basic chords"],
                "violin": ["Practice open strings"]
            },
            "intermediate": {
                "piano": ["Study counterpoint"],
                "guitar": ["Learn barre chords"],
                "violin": ["Practice shifting"]
            },
            "advanced": {
                "piano": ["Master advanced repertoire"],
                "guitar": ["Develop improvisation skills"],
                "violin": ["Explore extended techniques"]
            }
        }
        for level, instruments in defaults.items():
            for instr, rec_list in instruments.items():
                for rec in rec_list:
                     self.add_document(f"{level} {instr} practice: {rec}",
                                      metadata={'type': 'practice_tip', 'level': level, 'instrument': instr})
        return defaults

    def get_practice_recommendations(self, skill_level, instrument, focus_area=None):
        """Get recommendations using KnowledgeAgent retrieval"""
        query = f"{skill_level} {instrument} practice"
        if focus_area:
             query += f" focus on {focus_area}" # Enhance query if focus_area provided

        # Use KnowledgeAgent's retrieval instead of direct dict access
        results = self.retrieve(
            query,
            filters={'metadata.type': 'practice_tip'}
        )
        # Extract text from retrieved documents, limited to top 5 results
        return [doc['text'] for score, doc in results[:5]]

    def explain_concept(self, concept):
        """Explain a concept using ontology expansion"""
        # Retrieve information using ontology if available
        results = self.retrieve(concept, use_ontology=True, k=1)
        return results[0][1]['text'] if results else f"Concept '{concept}' not found."

    def explain_music_theory(self, concept):
        """Explain music theory concepts using retrieval with relevance"""
        # Check internal dict first for direct match
        if concept.lower() in self.music_theory_data:
            return self.music_theory_data[concept.lower()]

        # If not found, use retrieval
        results = self.retrieve(f"Explain music theory concept: {concept}", k=1)
        return results[0][1]['text'] if results else f"Music theory concept '{concept}' not found." # Return text of top result

    def get_musical_style(self, style):
        """Gets a description of a musical style"""
        style_lower = style.lower() # Check internal dict first
        if style_lower in self.style_data:
             return self.style_data[style_lower]
        # Fallback using retrieval
        results = self.retrieve(f"Describe the musical style: {style}", k=1)
        return results[0][1]['text'] if results else "Style not found in knowledge base"

    def handle_followup_question(self, previous_query, new_query):
        """Maintain session context for follow-up questions"""
        # Use contextual search for follow-ups
        results = self.contextual_search(
            f"{previous_query} {new_query}",
            context_window=3
        )
        return results[0][1]['text'] if results else "I couldn't find information related to that follow-up."

class Musician(QObject):
    """
    AI Musician model for creative assistance and learning.
    Integrates language understanding, music theory, reasoning, and output generation.
    """

    def __init__(self, shared_memory, agent_factory, expert_level=None, instrument=None, parent=None):
        super().__init__(parent)
        self.defer_setup = expert_level is None or instrument is None
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

        # --- Core modules ---
        self.transformer = Transformer(num_layers=6, embed_dim=512, num_heads=8, ff_dim=2048, num_styles=14)
        self.qnn_agent = QNNAgent(shared_memory, agent_factory, config={
            "num_qubits": 4,
            "num_quantum_layers": 2,
            "meta_learning_rate": 0.01,
            "qrnn_params": {"hidden_units": 16},
            "state_size": 128,
            "action_size": 128
        })
        self.theory_engine = MusicTheoryEngine()
        self.output_module = OutputModule()

        # --- Style and State ---
        self.current_style = None

        # Initialize core components, consistent with provided structure
        self.slailm = SLAILM(shared_memory, agent_factory)
        # Pass the specific tuple key expected by ReasoningAgent
        self.reasoner = ReasoningAgent(shared_memory, agent_factory, tuple_key="music|property|value")
        self.knowledge_base = KnowledgeBase(shared_memory, agent_factory)
        self.output_module = OutputModule()
        self.music_theory_engine = MusicTheoryEngine()
        self.transformer = None # Initialize lazily if needed
        self.qrnn = None # Initialize lazily if needed

        # User profile attributes
        self.expert_level = expert_level
        self.instrument = instrument

        # Internal state for music generation/editing
        self.current_tempo = 120 # Default tempo
        self.current_key = 'C major' # Default key
        self.generation_style = 'classical' # Default style
        self.current_goals = {} # To store user goals
        self.training_data_buffer = deque(maxlen=200) # Buffer for user edits

        # Setup user profile if needed
        self.interactive_setup()

    def interactive_setup(self):
        """ Fallback setup if expert level/instrument not provided initially """
        if self.defer_setup:
            # In a real application, this would use a GUI dialog
            logger.info("Deferring setup: Using default expert level (beginner) and instrument (piano)") #
            self.expert_level = "beginner"
            self.instrument = "piano"
            self.defer_setup = False # Mark setup as complete

    @pyqtSlot()
    def handle_generate_request(self):
        """ Slot for handling 'Generate' button clicks. """
        logger.info("Musician: Generate request received!")
        try:
            # Example: Generate a melody based on current state
            prompt = f"Generate a {self.generation_style} melody in {self.current_key} at {self.current_tempo} BPM for {self.instrument}" # Use current state
            # Call the generation function
            music_stream = self.generate_melody(prompt) # Assuming generate_melody returns a music21 stream
            if music_stream:
                 # Use output module to show/save the result
                 music_xml = music_stream.write('musicxml') # Convert stream to MusicXML string
                 self.output_module.generate_sheet_music_output(music_xml)
                 # Optional: Synthesize audio
                 # audio_data = self.output_module.convert_format(music_stream, 'wav')
                 # self.output_module.generate_audio_output(audio_data)
            else:
                 logger.warning("Melody generation failed.")

        except Exception as e:
            logger.error(f"Error during generation request: {e}") #

    @pyqtSlot(int)
    def handle_tempo_change(self, tempo_value):
        """ Slot for handling tempo slider changes. """
        if 40 <= tempo_value <= 300: # Validate tempo range
            self.current_tempo = tempo_value
            logger.info(f"Musician: Tempo updated to {self.current_tempo} BPM") #
            # Future: Adjust playback speed or trigger modifications if applicable
        else:
             logger.warning(f"Invalid tempo value received: {tempo_value}")

    @pyqtSlot(int)
    def handle_key_change(self, key_value):
        """ Slot for handling key slider changes. """
        try:
            # Convert slider value (0-11) to key name using music theory engine
            # Assuming key_value 0 = C, 1 = C#, ..., 11 = B
            key_name = self.music_theory_engine.REVERSE_NOTE_MAP.get(key_value % 12, 'C') # Use reverse map
            # For simplicity, assume major key for now, can be expanded
            self.current_key = f"{key_name} major"
            logger.info(f"Musician: Key updated to {self.current_key}")
        except Exception as e:
            logger.error(f"Error handling key change: {e}")

    @pyqtSlot(str)
    def handle_style_change(self, style):
        """ Slot for handling style changes (e.g., from a dropdown). """
        # Validate against known styles in knowledge base
        if style.lower() in self.knowledge_base.style_data:
            self.generation_style = style.lower()
            logger.info(f"Musician: Generation style set to {self.generation_style}")
        else:
             logger.warning(f"Unknown style selected: {style}")

    # --- Input Processing ---
    def process_input(self, input_data, input_type):
        """ Processes various musical input types. """
        logger.debug(f"Processing input type: {input_type}")
        try:
            if input_type == 'audio':
                # audio_data is expected to be a file path here based on analyze_audio_performance
                return self.process_audio(input_data)
            elif input_type == 'midi':
                 # midi_data could be path or mido object
                return self.process_midi(input_data)
            elif input_type == 'text':
                return self.process_text(input_data)
            elif input_type == 'sheet_music':
                 # sheet_music_data could be path or music21 object
                return self.process_sheet_music(input_data)
            else:
                raise ValueError(f"Invalid input type: {input_type}")
        except Exception as e:
            logger.error(f"Error processing input ({input_type}): {e}")
            return None # Return None or raise error based on desired handling

    def process_audio(self, audio_path):
        """ Loads and reshapes audio using AudioUtils. """
        logger.info(f"Processing audio file: {audio_path}")
        try:
            waveform = AudioUtils.load_audio(audio_path) # Use utility function
            reshaped_waveform = AudioUtils.reshape_for_model(waveform) # Use utility function
            return reshaped_waveform # Return the processed numpy array
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_path}")
            return None
        except Exception as e:
             logger.error(f"Error processing audio {audio_path}: {e}")
             return None

    def process_midi(self, midi_input):
        """ Loads MIDI data using mido. """
        logger.info("Processing MIDI data...")
        try:
            if isinstance(midi_input, str) or isinstance(midi_input, Path):
                midi_file = mido.MidiFile(midi_input) # Load from path
                return midi_file
            elif isinstance(midi_input, mido.MidiFile):
                 return midi_input # Already a mido object
            else:
                 # Attempt to parse if it's raw message list (less robust)
                 logger.warning("Processing raw MIDI message list - parsing may be limited.")
                 return list(midi_input) # Return the list as is
        except Exception as e:
            logger.error(f"Error processing MIDI: {e}")
            return None

    def process_text(self, text_data):
        """ Processes text using the SLAILM or intercepts explain requests with related concepts. """
        logger.info("Processing text data...")

        text_data = text_data.strip()

        if text_data.lower().startswith('explain'):
            try:
                # Extract concept after 'explain'
                concept_query = text_data[len('explain'):].strip().lower()

                # Load explanation
                explanation_dict = self.knowledge_base.explain_music_theory(concept_query)

                if isinstance(explanation_dict, dict):
                    level = self.expert_level.lower()
                    explanation_text = explanation_dict.get(level, "No explanation found for this level.")
                else:
                    explanation_text = explanation_dict

                # Load related concepts
                related_terms = []
                try:
                    with open('models/music/data/music_onthology.json', 'r', encoding='utf-8') as f:
                        ontology_data = json.load(f)
                    related_terms = ontology_data.get('nodes', {}).get('music_theory', {}).get(concept_query, {}).get('related_to', [])
                except Exception as e:
                    logger.error(f"Error loading related terms: {e}")

                related_text = ", ".join(related_terms) if related_terms else "N/A"

                result = (
                    f"<b>{concept_query.upper()}</b><br>"
                    f"<span style='color:dodgerblue;'>[Intermediate]:</span> {explanation_text}<br>"
                    f"<span style='color:dodgerblue; text-decoration: underline;'>[Related to]:</span> {related_text}<br>"
                )
                return {"output_text": result}

                #result = (f"[{concept_query}]\n"
                #        f"    [{level}]: {explanation_text}\n"
                #        f"    [related to]: {related_text}\n"
                #        f"    []")

                #return {"output_text": result}

            except Exception as e:
                logger.error(f"Error processing explain request: {e}")
                return {"output_text": "Failed to process explanation request."}

        else:
            # Fallback to SLAILM regular processing
            processed_result = self.slailm.process_input(prompt="musical_prompt", text=text_data)
            return processed_result

    def generate(self, prompt, task_data=None):
        """ Unified interface to match CollaborativeAgent for compatibility with PromptThread """
        if not isinstance(prompt, str):
            logger.error(f"Invalid prompt type passed to generate: {type(prompt)}")
            return "Error: Prompt must be a string."
        logger.info("Musician: generate() called.")
        try:
            result = self.process_text(prompt)
            if isinstance(result, dict) and "output_text" in result:
                return result["output_text"]  # <-- Only return the string!
            return str(result)
        except Exception as e:
            logger.error(f"Error in Musician.generate: {e}")
            return "Error during generation."

    def post_music_theory_explanation(self, explanation_dict, result):
        """Post a music theory explanation formatted nicely."""

        if isinstance(result, dict):
            concept = explanation_dict["concept_query"]
            level = explanation_dict["level"]
            description = explanation_dict["explanation_text"]
            related = explanation_dict["related_text"]

            # --- Create a container layout ---
            explanation_layout = QVBoxLayout()

            # --- Create a label for the concept explanation ---
            explanation_label = QLabel()
            explanation_label.setStyleSheet("""
                QLabel {
                    color: white;
                    font-size: 14px;
                    margin-bottom: 8px;
                }
            """)
            explanation_label.setText(f"""
            <div style="font-size: 1.5em; font-weight: bold;">{concept}</div>
            <div style="color: gold; font-weight: bold;">[{level}]</div>
            <div style="margin-top: 8px;">{description}</div>
            <div style="margin-top: 6px; font-style: italic;">[related to]: {related}</div>
            """)
            explanation_label.setTextFormat(Qt.RichText)
            explanation_label.setWordWrap(True)

            explanation_layout.addWidget(explanation_label)

            # --- Create the interactive IconButtons ---
            buttons_layout = QHBoxLayout()

            copy_button = IconButton("frontend/assets/icons/copy_icon.svg", "frontend/assets/icons/copy_icon1.svg")
            upvote_button = IconButton("frontend/assets/icons/upvote_icon.svg", "frontend/assets/icons/upvote_icon1.svg")
            downvote_button = IconButton("frontend/assets/icons/downvote_icon.svg", "frontend/assets/icons/downvote_icon1.svg")

            buttons_layout.addWidget(copy_button)
            buttons_layout.addWidget(upvote_button)
            buttons_layout.addWidget(downvote_button)

            explanation_layout.addLayout(buttons_layout)

            # --- Add the whole explanation layout into your output area ---
            container = QWidget()
            container.setLayout(explanation_layout)

            self.output_area_layout.addWidget(container)

    def copy_explanation_to_clipboard(self, concept, description, related):
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(f"{concept}\n   {description}\n   Related to - {related}")

    def process_sheet_music(self, sheet_music_input):
        """ Loads sheet music using music21. """
        logger.info("Processing sheet music data...")
        try:
            if isinstance(sheet_music_input, str) or isinstance(sheet_music_input, Path):
                 score = music21.converter.parse(sheet_music_input)
                 return score
            elif isinstance(sheet_music_input, music21.stream.Score) or isinstance(sheet_music_input, music21.stream.Part):
                 return sheet_music_input # Already a music21 object
            else:
                 raise TypeError("Unsupported sheet music input type")
        except music21.converter.ConverterException as e:
            logger.error(f"Error parsing sheet music: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing sheet music: {e}")
            return None

    def generate_music(self, style: str = "blues", measures: int = 8):
        self.current_style = style
        style_id = self._style_to_id(style)

        # Step 1: Create input seeds (chord progression and key)
        key = self.theory_engine.random_key()
        progression = self.theory_engine.generate_chord_progression_for_style(style, key)
        melody_seed = self.theory_engine.generate_melody(key=key, measures=measures)

        # Step 2: Encode melody seed as tensor input for Transformer
        melody_tensor = self._encode_melody(melody_seed)

        # Step 3: Transformer generation (style-conditioned)
        transformer_output = self.transformer.forward(melody_tensor, style_id=style_id)

        # Step 4: Quantum enhancement (optional)
        qnn_output, _ = self.qnn_agent._qrnn_forward([transformer_output])

        # Step 5: Decode back into notes
        notes_stream = self._decode_to_music21_stream(qnn_output)

        return notes_stream

    def batch_generate_music(self, style: str = "blues", measures: int = 8, batch_size: int = 8):
        """
        Generate multiple music sequences in parallel (batch processing).
        Args:
            style: Musical style
            measures: How long each piece should be
            batch_size: How many sequences to generate at once
        Returns:
            list of music21.stream.Stream (one per generated piece)
        """
        self.current_style = style
        style_id = self._style_to_id(style)

        # Step 1: Generate batch of seed melodies
        input_tensors = []
        for _ in range(batch_size):
            key = self.theory_engine.random_key()
            melody_seed = self.theory_engine.generate_melody(key=key, measures=measures)
            melody_tensor = self._encode_melody(melody_seed)
            input_tensors.append(melody_tensor)

        # Step 2: Stack into a single tensor
        input_batch = np.vstack(input_tensors)  # shape: (batch_size, seq_len, feature_dim)

        # Step 3: Forward pass through Transformer (batched)
        transformer_outputs = self.transformer.forward(input_batch, style_id=style_id)

        # Step 4: Quantum enhancement (optional, also batched)
        qnn_outputs, _ = self.qnn_agent._qrnn_forward([transformer_outputs[i] for i in range(batch_size)])

        # Step 5: Decode each sequence into music21 streams
        music_streams = []
        for output_seq in qnn_outputs:
            stream = self._decode_to_music21_stream(np.expand_dims(output_seq, axis=0))
            music_streams.append(stream)

        return music_streams

    def save_outputs(self, notes_stream, filename_prefix="composition"):
        """Save outputs in MIDI, sheet music (MusicXML), and Audio formats"""
        # Save sheet music
        self.output_module.generate_sheet_music_output(notes_stream.write('musicxml'), filename=f"{filename_prefix}_sheet")
        
        # Save MIDI
        self.output_module.generate_midi_output(notes_stream.write('midi'), filename=f"{filename_prefix}_midi")
        
        # Save Audio (via MIDI synthesis or optionally direct audio model later)
        self.output_module.generate_audio_output(notes_stream.write('wav'), filename=f"{filename_prefix}_audio")

    def _encode_melody(self, melody_stream):
        """
        Convert a music21 Stream (melody) to a properly structured input tensor for the Transformer:
        - (batch_size, sequence_length, feature_dim)
        - Features: pitch, duration, velocity
        """
    
        events = []
        for note in melody_stream.notesAndRests:
            if note.isRest:
                pitch = 0  # Special value for rests
            else:
                pitch = note.pitch.midi
    
            duration = int(note.quarterLength * 4)  # E.g., 1 quarter note = 4 ticks
            velocity = 64  # Default velocity for symbolic inputs (MIDI dynamic level, optional)
    
            events.append([pitch, duration, velocity])
    
        # Pad or trim to fixed sequence length (consistency with Transformer)
        target_seq_len = 64  # Adjustable: match your transformer's max sequence length
    
        if len(events) < target_seq_len:
            pad_length = target_seq_len - len(events)
            events += [[0, 0, 0]] * pad_length  # Pad with dummy tokens
        else:
            events = events[:target_seq_len]  # Truncate if too long
    
        tensor = np.array(events, dtype=np.float32)
        tensor = tensor[np.newaxis, :, :]  # Add batch dimension
    
        return tensor

    def _decode_to_music21_stream(self, output_sequence):
        """
        Convert Transformer or QNN outputs back into a music21 Stream:
        - Input: (batch_size, sequence_length, feature_dim=3)
        - Each event: [pitch, duration, velocity]
        """
    
        stream = music21.stream.Stream()
    
        for event in output_sequence[0]:  # Assume batch_size=1
            pitch = int(event[0])
            duration_ticks = int(event[1])
            velocity = int(event[2])
    
            # Ignore invalid events (e.g., empty padding)
            if duration_ticks <= 0:
                continue
    
            duration_quarters = max(0.25, duration_ticks / 4.0)  # Convert back from ticks to quarterLength
    
            if pitch == 0:
                # Rest event
                n = music21.note.Rest()
                n.quarterLength = duration_quarters
            else:
                # Note event
                n = music21.note.Note(pitch)
                n.quarterLength = duration_quarters
                # Store dynamics as a volume annotation (optional)
                n.volume.velocity = velocity
    
            stream.append(n)
    
        return stream

    def _style_to_id(self, style):
        """Map style name to internal numeric ID"""
        style_to_id_map = {
            "blues": 0,
            "jazz": 1,
            "pop": 2,
            "funk": 3,
            "classical": 4,
            "rock": 5,
            "hiphop": 6,
            "country": 7,
            "techno": 8,
            "reggae": 9,
            "metal": 10,
            "r&b": 11,
            "folk": 12,
            "latin": 13,
        }
        return style_to_id_map.get(style.lower(), 0)

    # --- Performance Analysis ---
    def analyze_performance(self, input_data, input_type, reference_data=None, reference_type=None):
        """ Analyzes performance (audio/MIDI) and provides feedback, optionally comparing to a reference. """
        if input_type not in ['audio', 'midi']:
            raise ValueError("Input type must be 'audio' or 'midi' for performance analysis.")

        processed_input = self.process_input(input_data, input_type)
        if processed_input is None:
            return {"error": f"Failed to process input {input_type}."}

        analysis_results = {}
        if input_type == 'audio':
            # Input data should be the path for librosa loading
            analysis_results = self.analyze_audio_performance(input_data)
        elif input_type == 'midi':
            # Input data should be the processed mido object or list
            analysis_results = self.analyze_midi_performance(processed_input)

        comparison_results = None
        if reference_data is not None and reference_type is not None:
            processed_reference = self.process_input(reference_data, reference_type)
            if processed_reference:
                comparison_results = self.compare_with_reference(processed_reference, analysis_results, reference_type, input_type) #
            else:
                 logger.warning("Failed to process reference data for comparison.")

        # Combine analysis and comparison for feedback generation
        combined_analysis = analysis_results.copy()
        if comparison_results:
             combined_analysis.update(comparison_results)

        feedback = self.generate_feedback(combined_analysis, input_type)
        return {"analysis": analysis_results, "comparison": comparison_results, "feedback": feedback}

    def analyze_audio_performance(self, audio_path):
        """ Analyzes audio features using librosa. """
        logger.info(f"Analyzing audio performance: {audio_path}")
        try:
            y, sr = librosa.load(audio_path, sr=AudioUtils.TARGET_SR) # Load with target SR
            # Pitch analysis using YIN algorithm
            pitches, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')) #
            times = librosa.times_like(pitches)
            valid_pitches = pitches[voiced_flag]

            # Rhythm analysis (tempo and beats)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            # Timbre analysis using MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            avg_mfcc = np.mean(mfccs, axis=1)

            # Dynamics (RMS energy)
            rms = librosa.feature.rms(y=y)[0]
            avg_rms = np.mean(rms)
            peak_rms = np.max(rms)

            return {
                "pitch_hz": valid_pitches.tolist() if valid_pitches.size > 0 else [],
                "pitch_times": times[voiced_flag].tolist() if voiced_flag.any() else [],
                "estimated_tempo": float(tempo),
                "beat_times": beat_times.tolist(),
                "average_timbre_mfcc": avg_mfcc.tolist(),
                "average_dynamics_rms": float(avg_rms),
                "peak_dynamics_rms": float(peak_rms),
            }
        except Exception as e:
            logger.error(f"Error analyzing audio performance {audio_path}: {e}")
            return {"error": "Audio analysis failed."}

    def analyze_midi_performance(self, midi_object):
        """ Analyzes note events from MIDI data (mido object). """
        logger.info("Analyzing MIDI performance...")
        notes = []
        current_time = 0
        active_notes = {} # Track active notes for duration calculation

        if not isinstance(midi_object, mido.MidiFile):
            logger.warning("Input is not a Mido MidiFile, analysis might be limited.")
            # Handle list of messages if possible (simple case)
            if isinstance(midi_object, list):
                for msg in midi_object:
                    if msg.type == 'note_on' and msg.velocity > 0:
                         notes.append({"note": msg.note, "velocity": msg.velocity, "time": msg.time}) #
                    # Simplified - doesn't calculate duration properly
                return {"notes": notes, "warning": "Duration calculation skipped for message list input."} #
            else:
                return {"error": "Unsupported MIDI input type for analysis."}


        # Proper analysis for Mido MidiFile
        for track in midi_object.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time # Accumulate delta time in seconds (requires tempo map)

                # Tempo map calculation needed here for accurate timing
                # For simplicity, using raw time deltas for now

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = {'velocity': msg.velocity, 'start_time': current_time}
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_info = active_notes.pop(msg.note)
                        duration = current_time - start_info['start_time']
                        notes.append({
                            "note": msg.note,
                            "velocity": start_info['velocity'], # Use velocity from note_on
                            "time": start_info['start_time'], # Onset time
                            "duration": duration # Calculated duration
                        })

        # Add tempo information if available
        tempo_changes = []
        for track in midi_object.tracks:
            current_time = 0
            for msg in track:
                 current_time += msg.time # Simplified time accumulation
                 if msg.is_meta and msg.type == 'set_tempo':
                       tempo_changes.append({'time': current_time, 'tempo_bpm': mido.tempo2bpm(msg.tempo)})

        return {"notes": notes, "tempo_changes": tempo_changes} #

    def compare_with_reference(self, reference_analysis, performance_analysis, reference_type, performance_type):
        """ Compares performance analysis against a reference. (Placeholder) """
        logger.info(f"Comparing {performance_type} performance against {reference_type} reference.")
        # This requires alignment algorithms (e.g., Dynamic Time Warping for audio/MIDI timing)
        # and feature comparison logic. Implementing this fully is complex.
        # Placeholder logic:
        timing_accuracy = random.uniform(0.5, 0.95) # Simulate comparison result
        pitch_accuracy = random.uniform(0.6, 0.98) # Simulate comparison result
        dynamics_accuracy = random.uniform(0.5, 0.9) # Simulate comparison result

        # Example: if comparing MIDI notes
        # Need to align note sequences and calculate differences in onset, duration, pitch, velocity
        # Example: if comparing audio features
        # Need to align features (e.g., pitches, MFCCs) using DTW and calculate distances

        logger.warning("Comparison logic is a placeholder and needs detailed implementation.")

        return {
            "timing_accuracy": timing_accuracy,
            "pitch_accuracy": pitch_accuracy,
            "dynamics_accuracy": dynamics_accuracy,
            "comparison_status": "Placeholder Comparison"
            } #

    def generate_feedback(self, combined_analysis_results, input_type):
        """ Generates textual feedback using SLAILM based on analysis and comparison. """
        logger.info(f"Generating feedback for {input_type} performance...")
        try:
            # Create a detailed prompt for the language model
            prompt = f"Provide constructive feedback for a musician based on the following {input_type} performance analysis:\n" #
            prompt += json.dumps(combined_analysis_results, indent=2)
            prompt += f"\nThe musician's experience level is {self.expert_level} and they play {self.instrument}."
            prompt += "\nFocus on areas for improvement in pitch, timing, dynamics, and timbre (if applicable). Mention positive aspects as well."
            if "comparison_status" in combined_analysis_results:
                 prompt += "\nThe performance was compared to a reference."

            # Use SLAILM's generation capability
            result = self.slailm.forward_pass({"raw_text": prompt})
            feedback_text = result.get("text", "Could not generate feedback.")

            # Add specific recommendations based on analysis
            recommendations = self.get_practice_recommendations(self.expert_level, self.instrument)
            if recommendations:
                 feedback_text += "\n\nSuggested practice areas:\n- " + "\n- ".join(random.sample(recommendations, min(len(recommendations), 2))) # Add 1-2 random recommendations

            return feedback_text
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            return "An error occurred while generating feedback."

    # --- Music Generation & Knowledge ---
    def generate_melody(self, prompt):
         """ Generates a melody using the Music Theory Engine. """
         logger.info(f"Generating melody for prompt: {prompt}")
         # Basic prompt parsing (can be enhanced by SLAILM)
         key = self.current_key
         measures = 4 # Default measures
         tempo = self.current_tempo
         instrument = self.instrument

         # Attempt to extract parameters from prompt (simple regex)
         match = re.search(r"(\d+)\s*measures", prompt, re.IGNORECASE)
         if match: measures = int(match.group(1))
         match = re.search(r"([A-G][#b]?\s*(major|minor))", prompt, re.IGNORECASE)
         if match: key = match.group(1)
         match = re.search(r"(\d+)\s*BPM", prompt, re.IGNORECASE)
         if match: tempo = int(match.group(1))

         try:
             melody_stream = self.music_theory_engine.generate_melody(
                 key=key,
                 measures=measures,
                 tempo=tempo,
                 instrument=instrument,
                 contour_shape=random.choice(['arch', 'valley', 'ascending', 'descending']) # Random contour for variety
             ) #
             return melody_stream
         except Exception as e:
              logger.error(f"Melody generation failed: {e}")
              return None

    def generate_chord_progression(self, prompt):
        """ Generates a chord progression using the Music Theory Engine. """
        logger.info(f"Generating chord progression for prompt: {prompt}")
        key = self.current_key
        num_chords = 4 # Default length
        progression_input = 'I-V-vi-IV' # Default progression

        # Basic prompt parsing
        match = re.search(r"(\d+)\s*chords", prompt, re.IGNORECASE)
        if match:
            num_chords = int(match.group(1))
            progression_input = None # Generate random if num_chords specified
        else:
            match = re.search(r"progression\s*like\s*([IViv\-,\s]+)", prompt, re.IGNORECASE)
            if match: progression_input = match.group(1).strip() #

        match = re.search(r"in\s*([A-G][#b]?\s*(major|minor))", prompt, re.IGNORECASE)
        if match: key = match.group(1)

        try:
            chord_symbols = self.music_theory_engine.generate_chord_progression(
                key=key,
                progression_input=progression_input,
                num_chords=num_chords
            ) #
            # Convert ChordSymbol objects to a playable stream (e.g., simple chords)
            score = music21.stream.Score()
            part = music21.stream.Part()
            for cs in chord_symbols:
                chord_obj = music21.chord.Chord(cs.pitches) # Create chord from pitches
                chord_obj.duration = music21.duration.Duration(1.0) # Assign default duration (e.g., quarter note)
                part.append(chord_obj)
            score.insert(0, part)
            return score
        except Exception as e:
             logger.error(f"Chord progression generation failed: {e}")
             return None

    def generate_sheet_music(self, prompt):
        """ Generates sheet music using SLAILM, Transformer, and QNNAgent (as per original code). """
        logger.info(f"Generating sheet music for prompt: {prompt}")
        try:
             # Initialize models if not already done (lazy loading)
             if self.transformer is None:
                  self.transformer = Transformer(num_layers=4)
             if self.qrnn is None:
                  # QNNAgent needs config - provide a default or load from somewhere
                  qnn_config = {"num_qubits": 4} # Example config
                  self.qrnn = QNNAgent(self.shared_memory, self.agent_factory, config=qnn_config)

             # Tokenize prompt (needs a tokenizer function - using basic split for now)
             def tokenize_prompt(p): return p.lower().split() # Placeholder tokenizer
             tokens = tokenize_prompt(prompt)

             # Forward pass through transformer (assuming input needs embedding first)
             # This part is underspecified - assuming Transformer can take token IDs or needs embedding layer
             # sequence = self.transformer.forward(tokens) # This likely needs an embedding step
             logger.warning("Sheet music generation requires proper token embedding before Transformer.")
             # Placeholder: generate random sequence for demonstration
             sequence_length = 50
             embedding_dim = self.transformer.layers[0]['attention'].q_proj.data.shape[0] if self.transformer.layers else 512
             sequence = np.random.rand(1, sequence_length, embedding_dim) # Batch size 1

             # Decode sequence to notes (needs a decoder function)
             def decode_music_tokens(seq): # Placeholder decoder
                 notes = []
                 for i in range(seq.shape[1]):
                      pitch_val = 60 + int(np.argmax(seq[0, i, :embedding_dim//2]) % 24) # Simple mapping
                      duration_val = 0.25 * (1 + int(np.argmax(seq[0, i, embedding_dim//2:]) % 4))
                      note = music21.note.Note(pitch_val)
                      note.duration = music21.duration.Duration(duration_val)
                      notes.append(note)
                 return notes

             generated_notes = decode_music_tokens(sequence)

             # Create score and return MusicXML
             score = music21.stream.Score()
             part = music21.stream.Part()
             part.append(generated_notes)
             score.insert(0, part)
             music_xml_output = score.write('musicxml')
             return music_xml_output # Return the XML string
        except Exception as e:
             logger.error(f"Sheet music generation failed: {e}")
             return None

    def get_practice_recommendations(self, skill_level, instrument, focus_area=None):
        """ Gets practice recommendations from the Knowledge Base. """
        logger.info(f"Getting practice recommendations for {skill_level} {instrument} (focus: {focus_area})") #
        try:
            recommendations = self.knowledge_base.get_practice_recommendations(skill_level, instrument, focus_area) #
            return recommendations if recommendations else ["No specific recommendations found."]
        except Exception as e:
             logger.error(f"Error getting practice recommendations: {e}")
             return ["Error retrieving recommendations."]

    def explain_music_theory(self, concept):
        """ Explains a music theory concept using the Knowledge Base. """
        logger.info(f"Explaining music theory concept: {concept}")
        try:
            explanation = self.knowledge_base.explain_music_theory(concept)
            return explanation if explanation else f"Could not find an explanation for '{concept}'."
        except Exception as e:
             logger.error(f"Error explaining music theory concept '{concept}': {e}")
             return "An error occurred while explaining the concept."

    def analyze_musical_style(self, piece_input, input_type):
        """ Analyzes musical style using features and reasoning. (Placeholder) """
        logger.info(f"Analyzing musical style of {input_type}...")
        processed_input = self.process_input(piece_input, input_type)
        if not processed_input:
            return "Could not process input for style analysis."

        # Extract relevant features (e.g., harmony, rhythm, instrumentation)
        # This requires more detailed analysis functions similar to performance analysis
        features = {}
        if input_type == 'audio':
             # Extract harmony (e.g., chroma features), tempo, timbre etc. from audio_path
             # features = self.extract_audio_style_features(piece_input) # Needs implementation
             pass
        elif input_type == 'midi':
             # Extract harmonic rhythm, melodic contours, velocity patterns etc. from midi_object
             # features = self.extract_midi_style_features(processed_input) # Needs implementation
             pass
        elif input_type == 'sheet_music':
             # Use music21 analysis tools on the score object
             # features = self.extract_score_style_features(processed_input) # Needs implementation
             pass

        logger.warning("Musical style feature extraction is not fully implemented.")

        # Use Reasoner to infer style based on extracted features and knowledge base
        # Example: Add facts like ('piece', 'has_tempo', 180), ('piece', 'uses_chord', 'G7')
        # self.reasoner.add_fact(...)
        # query_result = self.reasoner.query(('piece', 'is_style', '?'))
        # inferred_style = query_result[0][2] if query_result else "Unknown"

        # Fallback: Use KnowledgeBase style descriptions if features are simple
        # style_guess = "jazz" # Placeholder based on features
        # return self.knowledge_base.get_musical_style(style_guess)

        return "Musical style analysis placeholder - requires feature extraction and reasoning implementation."

    # --- Goal Setting & User Interaction ---
    def set_goals_and_track_progress(self, goals):
        """ Sets musical goals and tracks progress (Basic Implementation). """
        logger.info(f"Setting goals: {goals}")
        if isinstance(goals, list):
            for goal in goals:
                 if isinstance(goal, str):
                      goal_id = f"goal_{hash(goal)}"
                      self.current_goals[goal_id] = {"description": goal, "progress": 0.0, "status": "pending"}
                 else:
                      logger.warning(f"Skipping invalid goal format: {goal}")
        elif isinstance(goals, dict):
             # Assuming dict format is {goal_id: {description: ..., progress: ...}}
             self.current_goals.update(goals)
        else:
             logger.error("Invalid goals format. Provide list of strings or a dictionary.")
             return None

        logger.info(f"Current goals: {self.current_goals}")
        # Progress tracking would involve evaluating user performance against goals over time
        return self.current_goals

    def on_user_edit(self, original_data, modified_data, data_type):
        """ Captures user edits to potentially use for fine-tuning. """
        logger.debug(f"User edit captured ({data_type})")
        # Calculate a representation of the diff (depends heavily on data_type)
        diff = self.calculate_diff(original_data, modified_data, data_type)
        if diff is not None:
             edit_record = {
                 'original': original_data, # Store original (or its representation)
                 'modified': modified_data, # Store modified (or its representation)
                 'diff': diff, # Store the calculated difference
                 'type': data_type,
                 'timestamp': time.time()
             }
             self.training_data_buffer.append(edit_record) # Add to buffer
             # Optionally log to shared memory if needed by other agents
             # self.shared_memory.append("training_data", edit_record)

             # Trigger fine-tuning periodically if buffer is full or after N edits
             if len(self.training_data_buffer) >= 100: # Example threshold
                  self.trigger_finetuning()

    def calculate_diff(self, original, modified, data_type):
        """ Calculates a difference representation between original and modified data. (Placeholder) """
        # This needs specific implementation based on data type (text, MIDI, score)
        # Example for text: simple line diff
        # Example for MIDI/Score: more complex structural comparison
        logger.warning(f"Diff calculation for type '{data_type}' is a placeholder.")
        if isinstance(original, str) and isinstance(modified, str):
             return {"text_diff": f"Original length: {len(original)}, Modified length: {len(modified)}"} # Very basic diff
        return {"info": "Diff not implemented for this type"} # Placeholder

    def trigger_finetuning(self):
        """ Placeholder for initiating a fine-tuning process on the model(s). """
        logger.info("Fine-tuning trigger condition met. Initiating fine-tuning process (Placeholder)...")
        training_batch = list(self.training_data_buffer)
        self.training_data_buffer.clear() # Clear buffer after getting data
        # In a real system, this would:
        # 1. Format the training_batch data appropriately.
        # 2. Send the data to a training service or module.
        # 3. Potentially update the model weights (e.g., self.slailm, self.reasoner) after training.
        logger.info(f"Fine-tuning would use {len(training_batch)} edit examples.")
        # Example: Log data or send to another process
        # self.shared_memory.set("finetuning_batch", training_batch)

    def validate_musical_pattern(self, pattern_string, pattern_type='general'):
        """ Validates a musical pattern using the grammar processor. """
        logger.debug(f"Validating {pattern_type} pattern: {pattern_string}")
        try:
            # Assuming SLAILM's grammar processor has a method for music syntax
            # The original call included instrument - pass it if available
            is_valid = self.slailm.grammar_processor.validate_music_syntax(
                pattern_string,
                instrument=self.instrument
            )
            logger.info(f"Pattern validation result: {is_valid}")
            return is_valid
        except AttributeError:
             logger.error("SLAILM's grammar processor does not have 'validate_music_syntax' method.")
             return False # Indicate failure if method is missing
        except Exception as e:
             logger.error(f"Error during pattern validation: {e}")
             return False

    def plot_generated_sequence(self, output_sequence):
        pitches = [e[0] for e in output_sequence[0]]
        durations = [e[1] for e in output_sequence[0]]
        plt.figure(figsize=(12, 4))
        plt.plot(pitches, label='Pitch')
        plt.plot(durations, label='Duration')
        plt.legend()
        plt.title("Generated Sequence Overview")
        plt.show()

class IconButton(QLabel):
    def __init__(self, gold_icon_path, white_icon_path, size=24, parent=None):
        super().__init__(parent)
        self.gold_icon = QPixmap(gold_icon_path)
        self.white_icon = QPixmap(white_icon_path)
        self.setPixmap(self.gold_icon.scaled(size, size))
        self.setFixedSize(size + 4, size + 4)  # Allow hover effect space
        self.size = size
        self.setStyleSheet("background: transparent;")
        self.hover_anim = QPropertyAnimation(self, b"geometry")
        self.hover_anim.setDuration(150)

    def mousePressEvent(self, event):
        self.setPixmap(self.white_icon.scaled(self.size, self.size))
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setPixmap(self.gold_icon.scaled(self.size, self.size))
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        rect = self.geometry()
        self.hover_anim.stop()
        self.hover_anim.setStartValue(rect)
        self.hover_anim.setEndValue(rect.adjusted(-2, -2, 2, 2))  # Slightly enlarge
        self.hover_anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        rect = self.geometry()
        self.hover_anim.stop()
        self.hover_anim.setStartValue(rect)
        self.hover_anim.setEndValue(rect.adjusted(2, 2, -2, -2))
        self.hover_anim.start()
        super().leaveEvent(event)

if __name__ == "__main__":
    # If running this file directly for testing, you might need a QApplication
    # from PyQt5.QtWidgets import QApplication
    # import sys
    # app = QApplication(sys.argv)
    musician = Musician(shared_memory=None, agent_factory=None)
    # sys.exit(app.exec_())
