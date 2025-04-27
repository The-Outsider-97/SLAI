import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, Tuple

# Core musical dependencies
import music21
import librosa
import librosa.display
import mido

# Audio processing dependencies
from models.music.audio_utils import AudioUtils
from src.agents.perception.modules.transformer import Transformer

class OutputModule:
    """Comprehensive musical output generation and visualization system
    
    Features:
    - Multi-format musical output (MIDI, MusicXML, audio, text)
    - Interactive visualization system
    - Cross-modal output fusion
    - Quality preservation through lossless format handling
    - Batch processing capabilities

    Maintains consistency with:
    - PerceptionAgent's multi-modal fusion
    - LanguageAgent's text processing standards
    - Musician's music theory foundations
    - AudioEncoder's signal processing patterns
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {
            'default_output_dir': 'generated_output',
            'audio_format': 'wav',
            'midi_ticks_per_beat': 480,
            'visualization_dpi': 300,
            'max_batch_size': 32
        }
        
        self._create_output_structure()
        self.sample_rate = AudioUtils.TARGET_SR
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize transformer for cross-modal conversions
        self.transformer = Transformer(num_layers=4)

    def _create_output_structure(self):
        """Ensure consistent output directory structure"""
        base_dir = Path(self.config['default_output_dir'])
        dirs = ['midi', 'audio', 'sheet_music', 'visualizations']
        for d in dirs:
            (base_dir / d).mkdir(parents=True, exist_ok=True)

    def generate_sheet_music_output(
        self,
        musicxml_content: str,
        filename: Optional[str] = None,
        format: str = 'musicxml'
    ) -> Path:
        """
        Enhanced sheet music generation with quality preservation
        
        Args:
            musicxml_content: Valid MusicXML content as string
            filename: Optional output filename
            format: Output format (musicxml|pdf|midi)
            
        Returns:
            Path to generated file
        """
        try:
            score = music21.converter.parse(musicxml_content)
            output_dir = Path(self.config['default_output_dir']) / 'sheet_music'
            
            if not filename:
                filename = f"composition_{int(time.time())}"
                
            output_path = output_dir / f"{filename}.{format}"
            
            score.write(format, fp=output_path)
            self.logger.info(f"Generated sheet music: {output_path}")
            return output_path
            
        except music21.converter.ConverterException as e:
            self.logger.error(f"MusicXML parsing failed: {str(e)}")
            raise ValueError("Invalid MusicXML content") from e

    def generate_audio_output(
        self,
        audio_data: np.ndarray,
        filename: Optional[str] = None,
        format: Optional[str] = None
    ) -> Path:
        """
        Multi-format audio output generation with resampling
        
        Args:
            audio_data: Numpy array of audio samples
            filename: Optional base filename
            format: Output format (wav|mp3|ogg)
            
        Returns:
            Path to generated audio file
        """
        format = format or self.config['audio_format']
        output_dir = Path(self.config['default_output_dir']) / 'audio'
        
        if not filename:
            filename = f"audio_{int(time.time())}"
            
        output_path = output_dir / f"{filename}.{format}"
        
        try:
            AudioUtils.save_audio(audio_data, output_path, sr=self.sample_rate)
            self.logger.info(f"Generated audio file: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Audio generation failed: {str(e)}")
            raise RuntimeError("Audio output generation failed") from e

    def generate_midi_output(
        self,
        midi_data: Union[mido.MidiFile, list],
        filename: Optional[str] = None
    ) -> Path:
        """
        Flexible MIDI generation from multiple input types
        
        Args:
            midi_data: MidiFile object or list of MIDI messages
            filename: Optional output filename
            
        Returns:
            Path to generated MIDI file
        """
        output_dir = Path(self.config['default_output_dir']) / 'midi'
        
        if not filename:
            filename = f"composition_{int(time.time())}"
            
        output_path = output_dir / f"{filename}.mid"
        
        try:
            if isinstance(midi_data, mido.MidiFile):
                midi_data.save(output_path)
            else:
                midi = mido.MidiFile(ticks_per_beat=self.config['midi_ticks_per_beat'])
                track = mido.MidiTrack()
                track.extend(midi_data)
                midi.tracks.append(track)
                midi.save(output_path)
                
            self.logger.info(f"Generated MIDI file: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"MIDI generation failed: {str(e)}")
            raise RuntimeError("MIDI output generation failed") from e

    def visualize_musical_data(
        self,
        data: Union[np.ndarray, music21.stream.Stream, mido.MidiFile],
        visualization_type: str = 'spectrogram',
        figsize: Tuple[int, int] = (12, 6),
        **kwargs
    ) -> plt.Figure:
        """
        Multi-modal visualization system with automatic type detection
        
        Args:
            data: Input data (audio, MIDI, or Music21 stream)
            visualization_type: Type of visualization
            figsize: Figure dimensions
            
        Returns:
            Matplotlib figure object
        """
        plt.figure(figsize=figsize, dpi=self.config['visualization_dpi'])
        
        if isinstance(data, np.ndarray):
            self._visualize_audio(data, visualization_type, **kwargs)
        elif isinstance(data, music21.stream.Stream):
            self._visualize_notation(data, **kwargs)
        elif isinstance(data, mido.MidiFile):
            self._visualize_midi(data, **kwargs)
        else:
            raise ValueError("Unsupported data type for visualization")
            
        plt.tight_layout()
        return plt.gcf()

    def _visualize_audio(self, waveform: np.ndarray, vis_type: str = 'spectrogram', **kwargs):
        """Core audio visualization engine"""
        if vis_type == 'spectrogram':
            S = librosa.amplitude_to_db(
                np.abs(librosa.stft(waveform)),
                ref=np.max
            )
            librosa.display.specshow(
                S, 
                sr=self.sample_rate,
                x_axis='time',
                y_axis='log',
                **kwargs
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            
        elif vis_type == 'waveform':
            librosa.display.waveshow(waveform, sr=self.sample_rate)
            plt.title('Waveform')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')

    def _visualize_notation(self, stream: music21.stream.Stream):
        """Sheet music visualization using Music21"""
        stream.show()

    def _visualize_midi(self, midi: mido.MidiFile):
        """MIDI piano roll visualization"""
        notes = []
        for msg in midi:
            if msg.type == 'note_on':
                notes.append({
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'time': msg.time
                })
                
        plt.eventplot(
            [n['note'] for n in notes],
            orientation='horizontal',
            lineoffsets=1,
            linelengths=0.8
        )
        plt.title('MIDI Piano Roll')
        plt.ylabel('MIDI Note Number')
        plt.xlabel('Time (ticks)')

    def convert_format(
        self,
        input_data: Union[str, Path, np.ndarray, music21.stream.Stream],
        target_format: str,
        **kwargs
    ) -> Union[Path, np.ndarray]:
        """
        Advanced format conversion system with neural transcoding support
        
        Supported conversions:
        - Audio <-> MIDI (via neural transcription model)
        - MIDI <-> MusicXML (structural conversion)
        - MusicXML -> Audio (via synthesis)
        - Cross-modal embeddings -> All formats
        
        Maintains consistency with:
        - Musician's music21 integration
        - AudioEncoder's signal processing
        - PerceptionAgent's transformer architecture
        - LanguageAgent's error handling
        
        Args:
            input_data: Input file path, numpy array, or music21 stream
            target_format: One of ['midi', 'musicxml', 'wav', 'mp3', 'npz']
            
        Returns:
            Converted data in target format
        """
        # Validate input against project constants
        target_format = target_format.lower()
        supported_formats = {'midi', 'musicxml', 'wav', 'mp3', 'npz'}
        if target_format not in supported_formats:
            raise ValueError(f"Unsupported target format: {target_format}")
    
        # Determine input type using project-standard type checking
        input_type = self._detect_input_type(input_data)
        
        try:
            # Convert through intermediate representation using music21 where possible
            if input_type == 'audio' and target_format == 'midi':
                return self._audio_to_midi(input_data, **kwargs)
            elif input_type == 'midi' and target_format == 'musicxml':
                return self._midi_to_musicxml(input_data, **kwargs)
            elif input_type == 'musicxml' and target_format == 'midi':
                return self._musicxml_to_midi(input_data, **kwargs)
            elif input_type == 'musicxml' and target_format in {'wav', 'mp3'}:
                return self._musicxml_to_audio(input_data, target_format, **kwargs)
            elif input_type == 'midi' and target_format in {'wav', 'mp3'}:
                return self._midi_to_audio(input_data, target_format, **kwargs)
            else:
                raise NotImplementedError(f"Conversion {input_type}->{target_format} not implemented")
                
        except Exception as e:
            self.logger.error(f"Format conversion failed: {str(e)}")
            raise RuntimeError("Conversion failed") from e
    
    def _detect_input_type(self, input_data) -> str:
        """Consistent type detection with Musician's processing"""
        if isinstance(input_data, music21.stream.Stream):
            return 'musicxml'
        elif isinstance(input_data, (str, Path)):
            ext = Path(input_data).suffix.lower()[1:]
            if ext in {'mid', 'midi'}:
                return 'midi'
            elif ext == 'xml':
                return 'musicxml'
            elif ext in {'wav', 'mp3'}:
                return 'audio'
        elif isinstance(input_data, np.ndarray):
            return 'audio'
        raise ValueError("Unrecognized input type")
    
    def _audio_to_midi(self, audio_data: np.ndarray, **kwargs) -> Path:
        """
        Neural audio transcription using PerceptionAgent's architecture
        Integrates with AudioEncoder's signal processing
        """
        # Preprocess audio using project-standard AudioUtils
        processed_audio = AudioUtils.reshape_for_model(audio_data)
        
        # Load pretrained transcription model (compatible with PerceptionAgent weights)
        if not hasattr(self, 'transcription_model'):
            self.transcription_model = Transformer(num_layers=6)
            self.transcription_model.load_pretrained(
                self._load_conversion_weights('audio_to_midi')
            )
        
        # Extract note events through transformer
        note_events = self.transcription_model(processed_audio)
        
        # Convert to MIDI using Musician's MIDI handling patterns
        midi_file = mido.MidiFile()
        track = mido.MidiTrack()
        
        # Add tempo metadata from kwargs or default
        tempo = kwargs.get('tempo', 120)
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
        
        # Convert neural output to MIDI messages
        for onset, pitch, duration in note_events:
            track.append(mido.Message('note_on', note=pitch, time=onset))
            track.append(mido.Message('note_off', note=pitch, time=duration))
        
        midi_file.tracks.append(track)
        return self.generate_midi_output(midi_file)
    
    def _midi_to_musicxml(self, midi_data: Union[Path, mido.MidiFile], **kwargs) -> Path:
        """Structural conversion using music21's MIDI processing"""
        # Load MIDI using Musician's standard approach
        if isinstance(midi_data, Path):
            midi_data = mido.MidiFile(midi_data)

        # Convert to music21 stream with proper error handling
        try:
            stream = music21.midi.translate.midiFileToStream(midi_data)
            return self.generate_sheet_music_output(stream.write('musicxml'))
        except music21.midi.MidiException as e:
            raise ValueError("Invalid MIDI content") from e

    def _musicxml_to_midi(self, musicxml_data: Union[str, Path], **kwargs) -> Path:
        """High-fidelity conversion using music21's writer"""
        try:
            score = music21.converter.parse(musicxml_data)
            return self.generate_midi_output(score)
        except music21.converter.ConverterException as e:
            raise ValueError("Invalid MusicXML content") from e

    def _musicxml_to_audio(self, musicxml_data: Union[str, Path], target_format: str, **kwargs) -> Path:
        """End-to-end conversion via MIDI synthesis"""
        # Convert to MIDI first
        midi_path = self._musicxml_to_midi(musicxml_data)
        # Then synthesize to audio
        return self._midi_to_audio(midi_path, target_format, **kwargs)

    def _midi_to_audio(self, midi_data: Union[Path, mido.MidiFile], target_format: str, **kwargs) -> Path:
        """
        Neural audio synthesis using PerceptionAgent-compatible architecture
        Integrates with Musician's MIDI handling
        """
        # Load pretrained synthesis model
        if not hasattr(self, 'synthesis_model'):
            self.synthesis_model = Transformer(num_layers=6)
            self.synthesis_model.load_pretrained(
                self._load_conversion_weights('midi_to_audio')
            )

        # Convert MIDI to model input format
        midi_features = self._extract_midi_features(midi_data)

        # Generate audio waveform
        synthesized = self.synthesis_model(midi_features)

        # Post-process using AudioUtils
        processed_audio = AudioUtils.reshape_for_playback(synthesized)
        return self.generate_audio_output(processed_audio, format=target_format)

    def _load_conversion_weights(self, conversion_type: str) -> dict:
        """Weight loading compatible with PerceptionAgent's pretrained handling"""
        weights_path = Path(__file__).parent / 'conversion_weights' / f'{conversion_type}.npz'
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing conversion weights: {conversion_type}")

        return np.load(weights_path, allow_pickle=True)

    def batch_process(self, input_list: list, processing_func: callable, **kwargs) -> list:
        """
        Batch processing system with memory management
        
        Args:
            input_list: List of input items
            processing_func: Processing function to apply
            
        Returns:
            List of processed outputs
        """
        batch_size = min(self.config['max_batch_size'], len(input_list))
        results = []
        
        for i in range(0, len(input_list), batch_size):
            batch = input_list[i:i+batch_size]
            processed = [processing_func(item, **kwargs) for item in batch]
            results.extend(processed)

        return results

    def generate_text_output(self, text: str, filename: Optional[str] = None, format: str = 'txt') -> Path:
        """
        Enhanced text output with music-specific formatting

        Args:
            text: Text content to output
            filename: Optional output filename
            format: Output format (txt|md|pdf)

        Returns:
            Path to generated text file
        """
        output_dir = Path(self.config['default_output_dir'])
        filename = filename or f"output_{int(time.time())}"
        output_path = output_dir / f"{filename}.{format}"
        
        with open(output_path, 'w') as f:
            f.write(text)
            
        self.logger.info(f"Generated text output: {output_path}")
        return output_path

    @staticmethod
    def validate_output_path(path: Union[str, Path]) -> Path:
        """Consistent path validation across modules"""
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        return path
