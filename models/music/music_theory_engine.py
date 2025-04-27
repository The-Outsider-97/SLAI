
import re
import music21
import random
from collections import Counter
import traceback # Added for detailed error reporting

# --- Music Theory Engine Rewrite ---

class MusicTheoryEngine:
    """
    A comprehensive engine for music theory analysis, generation, and reasoning.

    Leverages the music21 library for robust musical object representation
    and analysis capabilities. Designed to assist with composition, arrangement,
    and understanding advanced musical concepts.
    """

    # --- Class Constants (Consolidated & Expanded) ---

    # Note mapping (MIDI-like, C=0) with enharmonic preference (sharps)
    NOTE_MAP = {
        'C': 0, 'B#': 0,
        'C#': 1, 'Db': 1,
        'D': 2,
        'D#': 3, 'Eb': 3,
        'E': 4, 'Fb': 4,
        'F': 5, 'E#': 5,
        'F#': 6, 'Gb': 6,
        'G': 7,
        'G#': 8, 'Ab': 8,
        'A': 9,
        'A#': 10, 'Bb': 10,
        'B': 11, 'Cb': 11
    }

    REVERSE_NOTE_MAP = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }

    # Interval definitions (semitones) - Can be expanded
    INTERVALS = {
        'P1': 0, 'm2': 1, 'M2': 2, 'm3': 3, 'M3': 4, 'P4': 5,
        'A4': 6, 'd5': 6, 'P5': 7, 'A5': 8, # Added Augmented 5th
        'm6': 8, 'M6': 9, 'd7': 9, # Added Diminished 7th
        'm7': 10, 'M7': 11, 'P8': 12,
        'm9': 13, 'M9': 14, # Added 9ths
    }

    # Chord Quality definitions (intervals from root using names from INTERVALS)
    CHORD_QUALITIES = {
        # Triads
        "major": ['P1', 'M3', 'P5'],
        "minor": ['P1', 'm3', 'P5'],
        "diminished": ['P1', 'm3', 'd5'],
        "augmented": ['P1', 'M3', 'A5'],
        "sus2": ['P1', 'M2', 'P5'],
        "sus4": ['P1', 'P4', 'P5'],
        # Sevenths
        "dominant7": ['P1', 'M3', 'P5', 'm7'],
        "major7": ['P1', 'M3', 'P5', 'M7'],
        "minor7": ['P1', 'm3', 'P5', 'm7'],
        "diminished7": ['P1', 'm3', 'd5', 'd7'], # d7 = M6 semitones = 9
        "half-diminished7": ['P1', 'm3', 'd5', 'm7'], # m7b5
        "minor-major7": ['P1', 'm3', 'P5', 'M7'],
        "augmented7": ['P1', 'M3', 'A5', 'm7'],
        "augmented-major7": ['P1', 'M3', 'A5', 'M7'],
        # Extended chords (examples)
        "major9": ['P1', 'M3', 'P5', 'M7', 'M9'],
        "dominant9": ['P1', 'M3', 'P5', 'm7', 'M9'],
        "minor9": ['P1', 'm3', 'P5', 'm7', 'M9'],
    }
    # Add aliases after the initial definition
    CHORD_QUALITIES["maj"] = CHORD_QUALITIES["major"]
    CHORD_QUALITIES["min"] = CHORD_QUALITIES["minor"]
    CHORD_QUALITIES["dim"] = CHORD_QUALITIES["diminished"]
    CHORD_QUALITIES["aug"] = CHORD_QUALITIES["augmented"]
    CHORD_QUALITIES[""] = CHORD_QUALITIES["major"] # Default triad if empty quality
    CHORD_QUALITIES["7"] = CHORD_QUALITIES["dominant7"] # Default 7th
    CHORD_QUALITIES["M7"] = CHORD_QUALITIES["major7"]
    CHORD_QUALITIES["m7"] = CHORD_QUALITIES["minor7"]
    CHORD_QUALITIES["dim7"] = CHORD_QUALITIES["diminished7"]
    CHORD_QUALITIES["m7b5"] = CHORD_QUALITIES["half-diminished7"]
    CHORD_QUALITIES["mM7"] = CHORD_QUALITIES["minor-major7"]
    CHORD_QUALITIES["aug7"] = CHORD_QUALITIES["augmented7"]
    CHORD_QUALITIES["maj9"] = CHORD_QUALITIES["major9"]
    CHORD_QUALITIES["dom9"] = CHORD_QUALITIES["dominant9"] # Common alias
    CHORD_QUALITIES["min9"] = CHORD_QUALITIES["minor9"]


    # Scale definitions (intervals from root in semitones)
    SCALE_PATTERNS = {
        # Major / Minor & Modes
        "major": [0, 2, 4, 5, 7, 9, 11],
        "natural_minor": [0, 2, 3, 5, 7, 8, 10],
        "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
        "melodic_minor": [0, 2, 3, 5, 7, 9, 11], # Ascending form
        "ionian": [0, 2, 4, 5, 7, 9, 11], # Same as major
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
        "lydian": [0, 2, 4, 6, 7, 9, 11],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "aeolian": [0, 2, 3, 5, 7, 8, 10], # Same as natural minor
        "locrian": [0, 1, 3, 5, 6, 8, 10],
        # Pentatonic
        "pentatonic_major": [0, 2, 4, 7, 9],
        "pentatonic_minor": [0, 3, 5, 7, 10],
        "pentatonic_blues": [0, 3, 5, 6, 7, 10], # Minor pentatonic + blue note
        # Other common scales
        "whole_tone": [0, 2, 4, 6, 8, 10],
        "chromatic": list(range(12)),
        "octatonic_diminished_wh": [0, 2, 3, 5, 6, 8, 9, 11], # Whole-half
        "octatonic_diminished_hw": [0, 1, 3, 4, 6, 7, 9, 10], # Half-whole
    }
    # Add aliases after initial definition
    SCALE_PATTERNS["minor"] = SCALE_PATTERNS["natural_minor"] # Default minor
    SCALE_PATTERNS["blues"] = SCALE_PATTERNS["pentatonic_blues"]

    # Rhythm Values (relative to quarter note = 1.0)
    BEAT_DURATIONS = {
        'whole': 4.0, 'half': 2.0, 'quarter': 1.0, 'eighth': 0.5,
        '16th': 0.25, '32nd': 0.125,
        'dotted-half': 3.0, 'dotted-quarter': 1.5, 'dotted-eighth': 0.75,
        'dotted-16th': 0.375,
        # Tuplets relative to their containing duration (e.g., eighth triplet = 3 in space of 2 eighths)
        'eighth-triplet': 1.0 / 3.0, # 3 in the space of 1 quarter
        '16th-triplet': 0.5 / 3.0,    # 3 in the space of 1 eighth
        'quarter-triplet': 2.0 / 3.0, # 3 in the space of 1 half
    }

    # --- Initialization ---

    def __init__(self):
        """Initializes the MusicTheoryEngine."""
        # Potential future state: current key, time signature, etc.
        self.current_key = music21.key.Key('C')
        self.current_time_signature = music21.meter.TimeSignature('4/4')
        # Initialize music21 environment (optional, good practice)
        self.env = music21.environment.UserSettings()
        # self.env['musicxmlPath'] = 'music/musescore/executable'


    # --- Core Conversion & Utility Methods ---

    def _note_name_to_midi(self, note_name):
        """Converts note name (e.g., 'C#4') to MIDI pitch number."""
        try:
            return music21.pitch.Pitch(note_name).midi
        except music21.pitch.PitchException:
            print(f"Warning: Invalid note name '{note_name}'")
            return None

    def _midi_to_note_name(self, midi_pitch, prefer_sharps=True):
        """Converts MIDI pitch number to note name."""
        try:
            p = music21.pitch.Pitch(midi=midi_pitch)
            # music21's default representation prefers flats for some keys,
            # override if sharps preferred
            if prefer_sharps and p.accidental and p.accidental.name == 'flat':
                 return p.getEnharmonic().nameWithOctave
            elif not prefer_sharps and p.accidental and p.accidental.name == 'sharp':
                 return p.getEnharmonic().nameWithOctave
            else:
                 return p.nameWithOctave
        except music21.pitch.PitchException:
            print(f"Warning: Invalid MIDI pitch '{midi_pitch}'")
            return None

    def _get_interval_semitones(self, interval_name):
        """Gets semitones for a named interval using internal map or music21."""
        # Basic interval lookup first
        if interval_name in self.INTERVALS:
            return self.INTERVALS[interval_name]
        # Fallback to music21 for complex/less common intervals
        try:
            iv = music21.interval.Interval(interval_name)
            # Store it for future use if found by music21
            self.INTERVALS[interval_name] = iv.semitones
            return iv.semitones
        except music21.interval.IntervalException:
             print(f"Warning: Unknown interval name '{interval_name}'")
             return None

    def _parse_root_and_quality(self, chord_name):
         """Parses 'C#m7' into ('C#', 'm7'). Handles complex names via music21."""
         try:
            # Use music21's harmony parsing
            cs = music21.harmony.ChordSymbol(chord_name)
            root = cs.root().name
            # Map music21 quality back to our keys if possible
            quality = cs.quality
            if quality == 'major' and not cs.isTriad(): quality = 'major7' # Adjust for M7
            # Add more specific mappings from music21 qualities to our keys...
            quality_map = {
                'major': 'major', 'minor': 'minor', 'diminished': 'diminished',
                'augmented': 'augmented', 'major-seventh': 'major7', 'minor-seventh': 'minor7',
                'dominant-seventh': 'dominant7', 'diminished-seventh': 'diminished7',
                'half-diminished': 'half-diminished7', # music21 uses 'half-diminished' for m7b5 figure 'ø'
                'minor-major-seventh': 'minor-major7',
                 # ... map other complex qualities if needed ...
            }
            mapped_quality = quality_map.get(quality, quality) # Keep original if no simple map

            # Check if mapped quality is in our list, otherwise try original
            if mapped_quality in self.CHORD_QUALITIES:
                 return root, mapped_quality
            elif quality in self.CHORD_QUALITIES:
                return root, quality
            else:
                # If still not found, maybe it's a simple alias
                if quality == 'dominant': return root, 'dominant7'
                # Return the quality name music21 found, even if not in our list
                # This allows getting notes even for unmapped qualities
                return root, quality

         except Exception: # music21 parsing failed, use basic regex fallback
             # Find split point between root (note name) and quality (suffix)
             match = re.match(r'([A-Ga-g][#b]?)(\S*)', chord_name)
             if match:
                  root, quality = match.groups()
                  # Try to validate root and quality
                  try:
                      music21.pitch.Pitch(root) # Validate root
                      # Use empty string for major triad if quality is empty
                      quality_to_check = quality if quality else ""
                      if quality_to_check in self.CHORD_QUALITIES:
                          return root, quality_to_check
                  except music21.pitch.PitchException:
                       pass # Invalid root

         print(f"Warning: Could not parse chord name '{chord_name}' reliably.")
         return None, None


    # --- Scale and Mode Methods ---

    def get_scale(self, root_note='C', scale_type='major', octave=4):
        """
        Returns a music21 Scale object for the specified root and type.

        Args:
            root_note (str): Root note name (e.g., 'C', 'Bb', 'F#').
            scale_type (str): Name of the scale (e.g., 'major', 'dorian', 'pentatonic_minor').
            octave (int): Starting octave for the root note.

        Returns:
            music21.scale.Scale: The requested scale object, or None if invalid type.
        """
        scale_type_key = scale_type.lower().replace(' ', '_') # Normalize key
        if scale_type_key not in self.SCALE_PATTERNS:
            print(f"Warning: Scale type '{scale_type}' (normalized: {scale_type_key}) not found in SCALE_PATTERNS.")
            return None

        try:
            # music21 has many built-in scales, prefer using them
            if scale_type_key == 'major' or scale_type_key == 'ionian':
                return music21.scale.MajorScale(f"{root_note}{octave}")
            elif scale_type_key == 'natural_minor' or scale_type_key == 'aeolian':
                return music21.scale.MinorScale(f"{root_note}{octave}", 'natural')
            elif scale_type_key == 'harmonic_minor':
                 return music21.scale.MinorScale(f"{root_note}{octave}", 'harmonic')
            elif scale_type_key == 'melodic_minor':
                 # music21 default melodic is ascending only, which matches our pattern
                 return music21.scale.MinorScale(f"{root_note}{octave}", 'melodic')
            elif scale_type_key == 'chromatic':
                # Specify direction for chromatic if needed, default ascending
                return music21.scale.ChromaticScale(f"{root_note}{octave}", direction='ascending')
            elif scale_type_key == 'whole_tone':
                 return music21.scale.WholeToneScale(f"{root_note}{octave}")
            elif scale_type_key == 'dorian':
                 return music21.scale.DorianScale(f"{root_note}{octave}")
            elif scale_type_key == 'phrygian':
                 return music21.scale.PhrygianScale(f"{root_note}{octave}")
            elif scale_type_key == 'lydian':
                 return music21.scale.LydianScale(f"{root_note}{octave}")
            elif scale_type_key == 'mixolydian':
                 return music21.scale.MixolydianScale(f"{root_note}{octave}")
            elif scale_type_key == 'locrian':
                 return music21.scale.LocrianScale(f"{root_note}{octave}")
            elif scale_type_key == 'pentatonic_major':
                 # music21 pentatonic requires specifying type
                 return music21.scale.MajorPentatonicScale(f"{root_note}{octave}")
            elif scale_type_key == 'pentatonic_minor':
                 return music21.scale.MinorPentatonicScale(f"{root_note}{octave}")
            elif scale_type_key == 'pentatonic_blues':
                  # music21 doesn't have a direct 'BluesScale', build from MinorPentatonic
                  minor_penta = music21.scale.MinorPentatonicScale(f"{root_note}{octave}")
                  blue_note_interval = music21.interval.Interval('A4') # Tritone from root
                  blue_note = minor_penta.tonic.transpose(blue_note_interval)
                  blues_pitches = minor_penta.getPitches() + [blue_note]
                  # Create a custom scale object
                  blues_scale = music21.scale.ConcreteScale(pitches=sorted(list(set(blues_pitches)))) # Ensure unique & sorted
                  blues_scale.name = f"{root_note} Blues"
                  return blues_scale
            # Add more built-in mappings... Lydian, Mixolydian etc. exist too
            else:
                # Create a custom scale using intervals if not directly supported or mapped
                print(f"Info: Creating custom scale for '{scale_type}' using defined intervals.")
                intervals = self.SCALE_PATTERNS[scale_type_key]
                # Use music21.interval.Interval for clarity in construction
                interval_objects = [music21.interval.Interval(semi) for semi in intervals]
                # Create based on pitch class intervals first (Abstract)
                # Then realize it starting from the specific root+octave
                tonic_pitch = music21.pitch.Pitch(f"{root_note}{octave}")
                scale_pitches = [tonic_pitch.transpose(iv) for iv in interval_objects]
                concrete_scale = music21.scale.ConcreteScale(pitches=scale_pitches)
                # Set a meaningful name
                concrete_scale.name = f"{root_note} {scale_type.replace('_',' ').title()}"
                return concrete_scale

        except Exception as e:
            print(f"Error creating scale '{root_note} {scale_type}': {e}")
            print(traceback.format_exc())
            return None

    def get_scale_notes(self, root_note='C', scale_type='major', num_octaves=1, start_octave=4):
        """
        Calculates the notes in a scale over a specified number of octaves.

        Args:
            root_note (str): The root note of the scale.
            scale_type (str): The type of scale (e.g., 'major', 'minor', 'dorian').
            num_octaves (int): How many octaves the scale should span.
            start_octave (int): The octave of the starting root note.

        Returns:
            list[str]: A list of note names (e.g., ['C4', 'D4', 'E4', ...]),
                       or an empty list if the scale type is invalid.
        """
        scale_obj = self.get_scale(root_note, scale_type, start_octave)
        if not scale_obj:
            return []

        try:
            start_pitch = music21.pitch.Pitch(f"{root_note}{start_octave}")
            # Calculate end pitch carefully to include the final octave root
            end_pitch = music21.pitch.Pitch(f"{root_note}{start_octave + num_octaves}")

            # Get pitches using the scale object's method
            # The scale object itself handles the intervals correctly
            pitches = scale_obj.getPitches(start_pitch, end_pitch)

            # Ensure the final root is included if num_octaves > 0 and wasn't captured
            if pitches[-1] < end_pitch:
                 final_root_in_scale = scale_obj.nextPitch(pitches[-1], direction=music21.scale.DIRECTION_ASCENDING)
                 # Add the final root only if it's the expected one and missing
                 if final_root_in_scale and final_root_in_scale == end_pitch:
                      pitches.append(final_root_in_scale)
                 # Or if the last pitch was already correct, we are good.
            elif pitches[-1] > end_pitch: # Remove overshoot if any
                 pitches = [p for p in pitches if p <= end_pitch]


            return [p.nameWithOctave for p in pitches]
        except Exception as e:
            print(f"Error getting scale notes: {e}")
            print(traceback.format_exc())
            return []

    # --- Chord Methods ---

    def get_chord(self, root_note='C', quality='major', octave=4, voicing=None):
        """
        Creates a music21 Chord object.

        Args:
            root_note (str): Root note name (e.g., 'C', 'Bb', 'F#') OR a full chord name ('Am7').
            quality (str, optional): Chord quality (e.g., 'major', 'm7', 'dim'). Ignored if root_note is a full name.
            octave (int, optional): Octave of the root note. Applied if root_note is just a name ('C'), ignored if it has octave ('C4').
            voicing (str | list[int], optional): Specifies voicing.
                - 'close': Standard close position voicing (default).
                - 'drop2', 'drop3': Drop voicings.
                - list[int]: Specific MIDI pitches for the chord.

        Returns:
            music21.chord.Chord: The chord object, or None if invalid.
        """
        # Try parsing root_note as a full chord symbol first
        parsed_root, parsed_quality = self._parse_root_and_quality(root_note)

        if parsed_root:
             root_name = parsed_root
             # Use parsed quality; default to 'major' if parsing yielded root but no quality (e.g. input 'C')
             quality_key = parsed_quality if parsed_quality is not None else ""
             # Try to find the octave from the original input if possible
             try:
                  octave = music21.pitch.Pitch(root_note).octave
             except:
                  # If original input didn't have octave, use the provided default
                  pass # Keep octave as the default value
        else:
             # Assume root_note is just the root name (potentially with octave)
             try:
                 p = music21.pitch.Pitch(root_note)
                 root_name = p.name
                 octave = p.octave
             except music21.pitch.PitchException:
                 # If not a pitch, assume it's just name like 'C'
                 root_name = root_note
                 # Keep octave as the default value

             # Use the provided quality argument
             quality_key = quality.lower() if quality else "" # Use "" for default major

        # Ensure quality_key is valid in our definitions
        if quality_key not in self.CHORD_QUALITIES:
            print(f"Warning: Chord quality '{quality_key}' (derived from '{quality}') not recognized in CHORD_QUALITIES.")
            # Attempt common aliases again if needed
            if quality_key == 'min': quality_key = 'minor'
            elif quality_key == 'maj': quality_key = 'major'
            elif quality_key == 'dom': quality_key = 'dominant7'
            # Fallback: Check if it's a quality music21 knows, even if we don't have intervals mapped
            elif parsed_quality and parsed_quality != quality_key:
                 quality_key = parsed_quality # Use the quality music21 parsed if different
            else:
                 print(f"Error: Cannot proceed with unknown quality '{quality_key}'.")
                 return None # Give up if still not found or derived

        try:
            # Get interval names for the quality
            interval_names = self.CHORD_QUALITIES.get(quality_key)

            # If quality_key was from music21 parsing but not in our list, we can't use our intervals
            if interval_names is None:
                 # Fallback: Try creating chord directly using music21 ChordSymbol
                 print(f"Info: Quality '{quality_key}' not in internal list, attempting music21 ChordSymbol directly.")
                 chord_symbol_str = f"{root_name}{quality_key}" # Reconstruct approx symbol
                 cs = music21.harmony.ChordSymbol(chord_symbol_str)
                 cs.root(music21.pitch.Pitch(f"{root_name}{octave}"))
                 chord_obj = music21.chord.Chord(cs.pitches) # Get pitches from ChordSymbol
                 chord_obj.commonName = cs.figure # Use figure as common name
            else:
                 # Build from our defined intervals
                 root_pitch = music21.pitch.Pitch(f"{root_name}{octave}")
                 pitches = [root_pitch] # Start with root
                 for interval_name in interval_names[1:]: # Skip P1
                      semitones = self._get_interval_semitones(interval_name)
                      if semitones is not None:
                          # Transpose root to get next pitch
                          pitches.append(root_pitch.transpose(semitones))
                      else:
                           print(f"Warning: Could not resolve interval '{interval_name}' for quality '{quality_key}'. Skipping.")

                 if not pitches:
                     print(f"Error: No valid pitches found for quality '{quality_key}'")
                     return None

                 chord_obj = music21.chord.Chord(pitches)
                 chord_obj.root(root_pitch) # Explicitly set root
                 # Try to set a common name using ChordSymbol parsing if possible
                 try:
                      cs_temp = music21.harmony.chordSymbolFromChord(chord_obj)
                      chord_obj.commonName = cs_temp.figure
                 except:
                      chord_obj.commonName = f"{root_name} {quality_key}" # Basic name fallback

            # Apply voicing if specified
            if voicing:
                chord_obj = self.apply_voicing(chord_obj, voicing)

            return chord_obj

        except Exception as e:
            print(f"Error creating chord '{root_name} {quality_key}' (original input: '{root_note}'): {e}")
            print(traceback.format_exc())
            return None

    def get_chord_notes(self, root_note='C', quality='major', octave=4):
        """
        Calculates the note names in a chord.

        Args:
            root_note (str): The root note of the chord (e.g., 'C4', 'G3').
                             Can also be a full chord name (e.g., 'Am7').
            quality (str, optional): The type of chord (e.g., 'major', 'm7', 'dim').
                                    Ignored if root_note is a full chord name.
            octave (int, optional): The octave of the root note. Used if not specified in root_note.

        Returns:
            list[str]: A list of note names in the chord (e.g., ['C4', 'E4', 'G4']).
                       Returns empty list if chord type is invalid.
        """
        # The get_chord method now handles parsing full names and octaves
        chord_obj = self.get_chord(root_note, quality, octave)
        if chord_obj:
            # Ensure notes are sorted by pitch for predictable output
            return [p.nameWithOctave for p in sorted(chord_obj.pitches)]
        else:
            return []

    def apply_voicing(self, chord_obj, voicing_type='close'):
         """
         Applies different voicings to a music21 Chord object.

         Args:
             chord_obj (music21.chord.Chord): The input chord.
             voicing_type (str | list[int]): 'close', 'drop2', 'drop3', or list of MIDI pitches.

         Returns:
             music21.chord.Chord: The re-voiced chord, or the original if voicing fails.
         """
         if not isinstance(chord_obj, music21.chord.Chord):
             print("Warning: apply_voicing requires a music21.chord.Chord object.")
             return chord_obj # Return original if not a chord

         # Ensure pitches are sorted for drop voicings
         pitches = sorted(chord_obj.pitches)

         try:
             if voicing_type == 'close':
                 # closedPosition finds the arrangement with smallest intervals
                 # forceOctave ensures the root stays in roughly the original octave area
                 return chord_obj.closedPosition(forceOctave=chord_obj.root().octave)
             elif voicing_type == 'drop2':
                 if len(pitches) < 2: return chord_obj # Need at least 2 notes
                 if len(pitches) < 4: print("Warning: Drop2 voicing ideally needs 4+ notes.")
                 # Take 2nd note from top and drop it by an octave
                 second_from_top = pitches[-2]
                 remaining_pitches = pitches[:-2] + [pitches[-1]] # All except 2nd from top
                 # Create new pitch object an octave lower
                 dropped_pitch = music21.pitch.Pitch(second_from_top.nameWithOctave)
                 dropped_pitch.octave -= 1
                 new_pitches = sorted(remaining_pitches + [dropped_pitch])
                 return music21.chord.Chord(new_pitches)
             elif voicing_type == 'drop3':
                  if len(pitches) < 3: return chord_obj # Need at least 3 notes
                  if len(pitches) < 4: print("Warning: Drop3 voicing ideally needs 4+ notes.")
                  # Take 3rd note from top and drop it by an octave
                  third_from_top = pitches[-3]
                  remaining_pitches = pitches[:-3] + pitches[-2:] # All except 3rd from top
                  dropped_pitch = music21.pitch.Pitch(third_from_top.nameWithOctave)
                  dropped_pitch.octave -= 1
                  new_pitches = sorted(remaining_pitches + [dropped_pitch])
                  return music21.chord.Chord(new_pitches)
             elif isinstance(voicing_type, list) and all(isinstance(p, int) for p in voicing_type):
                 # Assume list of MIDI pitches
                 voiced_pitches = [music21.pitch.Pitch(midi=p) for p in voicing_type]
                 # Create new chord, try to keep original root if possible
                 new_chord = music21.chord.Chord(voiced_pitches)
                 try: new_chord.root(chord_obj.root())
                 except: pass # Keep new root if original doesn't fit
                 return new_chord
             else:
                 print(f"Warning: Voicing type '{voicing_type}' not recognized. Returning original.")
                 return chord_obj
         except Exception as e:
              print(f"Error applying voicing '{voicing_type}': {e}")
              print(traceback.format_exc())
              return chord_obj # Return original on error


    # --- Analysis Methods ---

    def identify_harmony(self, notes):
        """
        Identifies the most likely chord or scale represented by a list of notes using music21.

        Args:
            notes (list[str | music21.pitch.Pitch | int]): List of note names ('C#4'),
                   music21 Pitch objects, or MIDI numbers.

        Returns:
            tuple(str, str): Best guess for (root, quality/scale_name) e.g., ('C', 'major'),
                           or ('Unknown', 'Unknown') if identification fails.
        """
        if not notes:
            return 'Unknown', 'Unknown'

        try:
            # Convert all input to unique music21 Pitch objects
            pitch_objects = set()
            for n in notes:
                try:
                    if isinstance(n, music21.pitch.Pitch):
                        pitch_objects.add(n)
                    elif isinstance(n, str):
                        pitch_objects.add(music21.pitch.Pitch(n))
                    elif isinstance(n, int):
                         pitch_objects.add(music21.pitch.Pitch(midi=n))
                    else:
                        print(f"Warning: Skipping unknown note type {type(n)} in identify_harmony.")
                except Exception as pitch_error:
                     print(f"Warning: Could not parse note '{n}' in identify_harmony: {pitch_error}")

            if not pitch_objects: return 'Unknown', 'Unknown'

            # --- Chord Identification ---
            # Use music21's chord identification with pitch objects
            chord_obj = music21.chord.Chord(list(pitch_objects))
            # analyze() provides methods like commonName, quality, etc.
            # chord_obj.analyse('key') # Could analyze potential key context if needed

            # Get the best guess for the chord name
            best_guess_name = chord_obj.pitchedCommonName # e.g., C4 major triad

            if best_guess_name:
                 # Try parsing this name back to get root and quality
                 root_guess, quality_guess = self._parse_root_and_quality(best_guess_name)
                 if root_guess and quality_guess:
                      # Map quality back to our keys if possible for consistency
                      quality_map = {
                          'major': 'major', 'minor': 'minor', 'diminished': 'diminished',
                          'augmented': 'augmented', 'major-seventh': 'major7', 'minor-seventh': 'minor7',
                          'dominant-seventh': 'dominant7', 'diminished-seventh': 'diminished7',
                          'half-diminished': 'half-diminished7',
                          'minor-major-seventh': 'minor-major7',
                          # ... add more mappings ...
                      }
                      mapped_quality = quality_map.get(quality_guess, quality_guess)
                      # Prefer our key if it exists
                      return root_guess, mapped_quality if mapped_quality in self.CHORD_QUALITIES else quality_guess

            # --- Scale Identification (Fallback) ---
            # If chord ID is uncertain or notes don't form a clear chord, try scales
            # music21 scale matching is complex, often requires more context (key, meter)
            # Basic check: see if notes fit a common scale pattern defined here
            pitch_classes = set(p.pitchClass for p in pitch_objects)
            num_unique_pcs = len(pitch_classes)

            possible_scales = {} # Store potential scale matches: {('Root', 'scale_name'): confidence}

            # Iterate through known scale patterns
            for scale_name, pattern_intervals in self.SCALE_PATTERNS.items():
                # Check if the number of unique pitch classes matches the scale size
                # This is a simple heuristic, not foolproof
                if num_unique_pcs >= len(pattern_intervals) / 2 and num_unique_pcs <= len(pattern_intervals) : # Allow partial matches
                    # Try each unique pitch class as a potential root
                    for root_pc in pitch_classes:
                        # Generate the pitch classes of the scale starting from this root
                        scale_pcs = set((root_pc + i) % 12 for i in pattern_intervals)
                        # Calculate how many input pitch classes are in this scale
                        match_count = len(pitch_classes.intersection(scale_pcs))
                        # Calculate confidence (e.g., proportion of matched notes)
                        confidence = match_count / num_unique_pcs if num_unique_pcs > 0 else 0
                        # Require a reasonable match (e.g., > 70% of notes fit)
                        if confidence > 0.7:
                             scale_key = (self.REVERSE_NOTE_MAP[root_pc], scale_name)
                             # Store the highest confidence found for this scale
                             possible_scales[scale_key] = max(possible_scales.get(scale_key, 0), confidence)

            # Find the best scale match based on confidence
            if possible_scales:
                best_scale, best_confidence = max(possible_scales.items(), key=lambda item: item[1])
                # Add a condition: only return scale if confidence is high enough and chord ID failed
                if best_confidence > 0.8 and not best_guess_name: # Threshold for returning scale
                     return best_scale[0], best_scale[1]


            # --- Final Fallbacks ---
            if len(pitch_objects) == 1:
                return list(pitch_objects)[0].name, 'Single Note'

            # If music21 chord name failed and scale match is weak/absent
            # Return the raw pitch classes as complex harmony
            pc_names = sorted([self.REVERSE_NOTE_MAP[pc] for pc in pitch_classes])
            return f"PCs: {','.join(pc_names)}", 'Complex Harmony'

        except Exception as e:
            print(f"Error identifying harmony: {e}")
            print(traceback.format_exc())
            return 'Error', 'Error'

    def analyze_harmonic_function(self, progression, key='C'):
        """
        Analyzes the harmonic function (e.g., Tonic, Dominant) of chords in a progression
        relative to a given key, using music21's RomanNumeral analysis.

        Args:
            progression (list[str | music21.chord.Chord | music21.harmony.ChordSymbol | music21.roman.RomanNumeral]):
                       List of chord names ('Am'), Chord objects, ChordSymbol objects, or RomanNumeral objects.
            key (str | music21.key.Key): The key context (e.g., 'C', 'Am', 'Bb major').

        Returns:
            list[str]: List of harmonic functions (e.g., ['T', 'D', 'T'], ['SD', 'D', 'T'])
                       or None if analysis fails. Functions: T=Tonic, D=Dominant, SD=Subdominant,
                       PD=Predominant (general term for SD/other chords leading to D),
                       TD=Tonic Expansion (e.g., iii, vi), SecD=Secondary Dominant,
                       Sub=Substitution (e.g., Tritone Sub), Mod=Modulatory Chord, Other=Chromatic/Non-functional.
        """
        try:
            if isinstance(key, str):
                try: key_obj = music21.key.Key(key)
                except: key_obj = music21.key.Key(key.split()[0], key.split()[1].lower() if len(key.split())>1 else 'major') # More robust key parsing
            elif isinstance(key, music21.key.Key):
                key_obj = key
            else:
                print("Warning: Invalid key format.")
                return None

            functions = []
            last_rn = None # Keep track of previous chord for context

            for i, chord_input in enumerate(progression):
                rn = None
                chord_obj_for_analysis = None

                # --- Convert input to RomanNumeral ---
                if isinstance(chord_input, music21.roman.RomanNumeral):
                    # Ensure it has the correct key context
                    if rn.key != key_obj:
                        try: rn = music21.roman.RomanNumeral(chord_input.figure, key_obj)
                        except: rn = chord_input # Keep original if re-parsing fails
                    else: rn = chord_input
                    chord_obj_for_analysis = music21.chord.Chord(rn.pitches) # Get chord for extra checks
                elif isinstance(chord_input, music21.chord.Chord):
                    chord_obj_for_analysis = chord_input
                    try: rn = music21.roman.romanNumeralFromChord(chord_input, key_obj)
                    except Exception as e_rn: print(f"Debug RN From Chord err: {e_rn}") # Keep going
                elif isinstance(chord_input, music21.harmony.ChordSymbol):
                    chord_obj_for_analysis = chord_input # Can treat ChordSymbol like Chord for RN analysis
                    try: rn = music21.roman.romanNumeralFromChord(chord_input, key_obj)
                    except Exception as e_rn: print(f"Debug RN From ChordSymbol err: {e_rn}")
                elif isinstance(chord_input, str):
                    # Try parsing as RomanNumeral first (e.g., 'V7/IV')
                    try:
                         rn = music21.roman.RomanNumeral(chord_input, key_obj)
                         chord_obj_for_analysis = music21.chord.Chord(rn.pitches)
                    except:
                         # If RN fails, try parsing as ChordSymbol (e.g., 'Am7')
                         try:
                              cs = music21.harmony.ChordSymbol(chord_input)
                              chord_obj_for_analysis = cs
                              rn = music21.roman.romanNumeralFromChord(cs, key_obj)
                         except Exception as e_final_parse:
                              print(f"Warning: Could not parse '{chord_input}' as Chord or RomanNumeral in key '{key_obj.name}': {e_final_parse}")
                              functions.append('Unknown')
                              last_rn = None
                              continue
                else:
                     print(f"Warning: Skipping unknown chord type {type(chord_input)}")
                     functions.append('Unknown')
                     last_rn = None
                     continue

                # --- Assign Harmonic Function based on RomanNumeral ---
                current_function = 'Other' # Default
                if rn:
                    # Basic function mapping (can be much more nuanced based on context)
                    scale_degree = rn.scaleDegree
                    quality = rn.quality # e.g., 'major', 'minor', 'dominant-seventh'
                    figure = rn.figure # e.g., 'V7', 'ii', 'vii°'

                    # 1. Diatonic Function Categories
                    if scale_degree == 1: current_function = 'T'
                    elif scale_degree == 5: current_function = 'D'
                    elif scale_degree == 4: current_function = 'SD'
                    elif scale_degree == 2: current_function = 'PD' # Predominant (often interchangeable with SD)
                    elif scale_degree == 7:
                         # Leading tone chord - function depends on quality
                         if quality in ['diminished', 'half-diminished', 'diminished-seventh']:
                              current_function = 'D' # Dominant function (leading to T)
                         else: # VII in minor or chromatic alterations
                              current_function = 'D' if figure == 'VII' else 'Other' # VII in minor can be subtonic (PD function) or Dom Sub? Let's simplify to 'Other' if not diminished
                    elif scale_degree == 3: current_function = 'TD' # Mediant (often Tonic Expansion/Substitution)
                    elif scale_degree == 6: current_function = 'TD' # Submediant (often Tonic Expansion/Substitution)

                    # 2. Secondary Functions / Chromaticism
                    if rn.secondaryRomanNumeral: # Checks for V/ii, V/V etc.
                         current_function = 'SecD'
                    elif rn.isNeapolitan():
                         current_function = 'Sub' # Neapolitan chord (Predominant Substitution)
                    elif rn.isItalianAugmentedSixth() or rn.isGermanAugmentedSixth() or rn.isFrenchAugmentedSixth():
                         current_function = 'Sub' # Augmented Sixths (Predominant Substitution)

                    # 3. Contextual Refinements (Examples - can be expanded significantly)
                    # - V7 resolving to something other than I might be 'Deceptive'
                    if last_rn and last_rn.scaleDegree == 5 and scale_degree != 1:
                         # Could refine D based on resolution, but let's keep it simple for now
                         pass
                    # - Cadential 6/4 (I64 before V) is Dominant function
                    if figure.endswith('64') and scale_degree == 1:
                         # Check if next chord is V
                         next_chord_input = progression[i+1] if (i+1) < len(progression) else None
                         if next_chord_input:
                              # Simplified check: if next chord is V, treat I64 as D
                              try:
                                   next_rn = music21.roman.RomanNumeral(str(next_chord_input), key_obj) # Basic parsing for check
                                   if next_rn.scaleDegree == 5:
                                        current_function = 'D' # Cadential 6/4 has Dominant function
                              except: pass # Ignore if next chord parsing fails

                    # 4. Check for Modulation Indication (less trivial)
                    # - If a chord strongly implies a new key (e.g., pivot chord analysis needed)
                    # - This requires more sophisticated analysis beyond single chord function
                    # - For now, we don't explicitly detect 'Mod'

                else:
                    # If RomanNumeral analysis failed entirely
                     current_function = 'Unknown'


                functions.append(current_function)
                last_rn = rn # Store for next iteration's context

            return functions

        except Exception as e:
            print(f"Error analyzing harmonic function: {e}")
            print(traceback.format_exc())
            return None

    def analyze_melodic_contour(self, notes):
        """
        Provides a simple analysis of melodic direction using MIDI pitch differences.

        Args:
            notes (list[str | music21.pitch.Pitch | int]): Sequence of notes/pitches. Can include rests (as None or 'R').

        Returns:
            str: General contour description (e.g., 'Ascending', 'Descending',
                 'Ascending-Descending', 'Static', 'Undulating', 'Mixed').
        """
        if len(notes) < 2:
            return 'Static' # Not enough notes for contour

        try:
            midi_pitches = []
            # Extract MIDI pitches, skipping rests
            for n in notes:
                 p = None
                 try:
                     if isinstance(n, music21.pitch.Pitch): p = n.midi
                     elif isinstance(n, music21.note.Note): p = n.pitch.midi # Handle Note objects
                     elif isinstance(n, str) and n.upper() != 'R': p = music21.pitch.Pitch(n).midi
                     elif isinstance(n, int): p = n
                     # Add valid pitches to list
                     if p is not None: midi_pitches.append(p)
                 except Exception:
                      pass # Ignore parsing errors or rests

            if len(midi_pitches) < 2: return 'Static' # Not enough pitched notes

            # Calculate differences between consecutive pitches
            diffs = [midi_pitches[i] - midi_pitches[i-1] for i in range(1, len(midi_pitches))]

            if not diffs: return 'Static' # Only one pitch after filtering rests

            ups = sum(1 for d in diffs if d > 0)
            downs = sum(1 for d in diffs if d < 0)
            sames = sum(1 for d in diffs if d == 0)

            total_intervals = len(diffs)

            # Simple cases first
            if ups == total_intervals: return 'Ascending'
            if downs == total_intervals: return 'Descending'
            if sames == total_intervals: return 'Static'
            if ups > 0 and downs == 0: return 'Mostly Ascending' # Includes static intervals
            if downs > 0 and ups == 0: return 'Mostly Descending' # Includes static intervals

            # More complex shapes (Arch/Valley detection)
            mid_point = total_intervals // 2
            first_half_diffs = diffs[:mid_point]
            second_half_diffs = diffs[mid_point:]

            first_half_ups = sum(1 for d in first_half_diffs if d > 0)
            first_half_downs = sum(1 for d in first_half_diffs if d < 0)
            second_half_ups = sum(1 for d in second_half_diffs if d > 0)
            second_half_downs = sum(1 for d in second_half_diffs if d < 0)

            # Arch: Mostly up then mostly down
            if (first_half_ups >= first_half_downs) and (second_half_downs > second_half_ups):
                 # Stronger check: last note lower than peak, first note lower than peak
                 peak_index = midi_pitches.index(max(midi_pitches))
                 if midi_pitches[-1] < midi_pitches[peak_index] and midi_pitches[0] < midi_pitches[peak_index]:
                     return 'Arch'

            # Valley: Mostly down then mostly up
            if (first_half_downs > first_half_ups) and (second_half_ups >= second_half_downs):
                 # Stronger check: last note higher than trough, first note higher than trough
                 trough_index = midi_pitches.index(min(midi_pitches))
                 if midi_pitches[-1] > midi_pitches[trough_index] and midi_pitches[0] > midi_pitches[trough_index]:
                     return 'Valley'

            # If none of the specific shapes match, call it undulating or mixed
            if ups > 0 and downs > 0: return 'Undulating' # General up and down movement

            return 'Mixed' # Default if other categories don't fit well

        except Exception as e:
            print(f"Error analyzing melodic contour: {e}")
            print(traceback.format_exc())
            return 'Error'

    # --- Generation Methods ---

    def generate_chord_progression(self, key='C major', progression_input='I-IV-V-I', num_chords=None):
        """
        Generate diatonic chord progressions using Roman numerals in a given key.
        Can generate from a specified pattern or create a random diatonic one.

        Args:
            key (str | music21.key.Key): Musical key (e.g., 'C Major', 'A Minor').
            progression_input (str | list[str]): Roman numeral progression (e.g., 'I-V-vi-IV')
                                                  or list of numerals ['I', 'V', 'vi', 'IV'].
                                                  Used if num_chords is None.
            num_chords (int, optional): If provided, generates a random diatonic
                                        progression of this length, ignoring `progression_input`.

        Returns:
            list[music21.harmony.ChordSymbol]: List of music21 ChordSymbol objects
                                               representing the progression. Returns empty list on error.
        """
        try:
            # --- Setup Key ---
            if isinstance(key, str):
                 try: key_obj = music21.key.Key(key)
                 except: key_obj = music21.key.Key(key.split()[0], key.split()[1].lower() if len(key.split())>1 else 'major')
            elif isinstance(key, music21.key.Key):
                key_obj = key
            else:
                print("Error: Invalid key format for progression generation.")
                return []

            progression_numerals = []

            # --- Determine Numerals: Random or Specified ---
            if num_chords is not None and num_chords > 0:
                # Generate Random Diatonic Progression
                # Define typical diatonic chords (can customize based on minor type etc.)
                major_numerals = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']
                # Natural minor default, but often V (major dom) from harmonic is used
                minor_numerals = ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII'] # Natural
                harmonic_minor_V = 'V' # Major dominant often borrowed

                possible_numerals = major_numerals if key_obj.mode == 'major' else minor_numerals
                # Add harmonic minor V if in minor key
                if key_obj.mode == 'minor' and harmonic_minor_V not in possible_numerals:
                     # Decide whether to replace 'v' or just add 'V'
                     # Replacing v with V is common practice for stronger cadences
                     try: possible_numerals[possible_numerals.index('v')] = harmonic_minor_V
                     except ValueError: possible_numerals.append(harmonic_minor_V) # Add if 'v' wasn't there


                # Simple random generation (can be improved with transition probabilities)
                # Start on Tonic
                current_numeral = 'I' if key_obj.mode == 'major' else 'i'
                progression_numerals = [current_numeral]
                for _ in range(num_chords - 1):
                    # Basic transition logic (example)
                    last_numeral_fig = current_numeral.split('/')[0] # Ignore secondary parts for simple rules
                    next_options = possible_numerals[:] # Copy list

                    if last_numeral_fig in ['V', 'V7', 'vii°']: next_options = ['I', 'i'] # Strong pull to tonic
                    elif last_numeral_fig in ['IV', 'iv', 'ii', 'ii°']: next_options = ['V', 'V7', 'vii°'] + next_options # Pull to dominant, allow others
                    elif last_numeral_fig in ['I', 'i']: next_options = possible_numerals # Any chord can follow tonic

                    # Remove the last chord to avoid immediate repetition (optional)
                    if current_numeral in next_options and len(next_options) > 1:
                         next_options.remove(current_numeral)

                    next_numeral = random.choice(next_options)
                    progression_numerals.append(next_numeral)
                    current_numeral = next_numeral

            elif isinstance(progression_input, str):
                # Split string by '-' or common delimiters like space, comma
                progression_numerals = re.split(r'[\s,-]+', progression_input)
                progression_numerals = [n for n in progression_numerals if n] # Remove empty strings
            elif isinstance(progression_input, list):
                progression_numerals = progression_input
            else:
                 print("Warning: Invalid progression_input format. Provide string or list.")
                 return []

            # --- Convert Numerals to Chord Symbols ---
            chord_symbols = []
            if not progression_numerals:
                 print("Warning: No Roman numerals determined for progression.")
                 return []

            for numeral_str in progression_numerals:
                if not isinstance(numeral_str, str) or not numeral_str.strip():
                     print(f"Warning: Skipping invalid numeral '{numeral_str}'")
                     continue
                try:
                    # Create RomanNumeral object first
                    rn = music21.roman.RomanNumeral(numeral_str.strip(), key_obj)
                    # Convert RomanNumeral to ChordSymbol for output
                    # Creating ChordSymbol directly from RN figure is often best
                    chord_symbol = music21.harmony.ChordSymbol(f"{rn.root().name}{rn.figure}")
                    # Set pitches based on RN for accuracy if needed later
                    # chord_symbol.pitches = rn.pitches
                    chord_symbols.append(chord_symbol)
                except Exception as e_parse_rn:
                    print(f"Warning: Could not parse Roman numeral '{numeral_str}' in key '{key_obj.name}': {e_parse_rn}")
                    # Optionally add a placeholder or skip
                    # chord_symbols.append(None)

            return chord_symbols

        except Exception as e_prog_gen:
            print(f"Error generating chord progression: {e_prog_gen}")
            print(traceback.format_exc())
            return []


    def generate_melody(self, key='C Major', measures=4, time_signature='4/4',
                          tempo=120, rhythm_complexity=0.5, contour_shape='arch',
                          instrument=None):
        """
        Generates a simple melody within a key using music21 streams.

        Args:
            key (str | music21.key.Key): Musical key.
            measures (int): Number of measures for the melody.
            time_signature (str | music21.meter.TimeSignature): Time signature.
            tempo (int): Tempo in BPM.
            rhythm_complexity (float): Value between 0 (simple quarter/eighth) and 1 (more syncopation, rests, varied durations).
            contour_shape (str): Desired shape ('arch', 'valley', 'ascending', 'descending', 'static', 'random').
            instrument (str | music21.instrument.Instrument, optional): Instrument name or object for the part.

        Returns:
            music21.stream.Part: A music21 Part stream containing the melody, or None on error.
        """
        try:
            # --- Setup ---
            if isinstance(key, str): key_obj = music21.key.Key(key)
            else: key_obj = key
            if isinstance(time_signature, str): ts_obj = music21.meter.TimeSignature(time_signature)
            else: ts_obj = time_signature

            melody_part = music21.stream.Part(id='melody_part')

            # Add Instrument if specified
            if instrument:
                 if isinstance(instrument, str):
                      try: melody_part.insert(0, music21.instrument.instrumentFromName(instrument))
                      except: print(f"Warning: Could not find instrument '{instrument}'.")
                 elif isinstance(instrument, music21.instrument.Instrument):
                      melody_part.insert(0, instrument)

            melody_part.append(music21.tempo.MetronomeMark(number=tempo))
            melody_part.append(key_obj)
            melody_part.append(ts_obj)

            # Define pitch range (e.g., 2 octaves around tonic)
            tonic_pitch = key_obj.tonic
            start_pitch_range = music21.pitch.Pitch(f'{tonic_pitch.name}3') # Example low end
            end_pitch_range = music21.pitch.Pitch(f'{tonic_pitch.name}5')   # Example high end

            scale_obj = self.get_scale(tonic_pitch.name, key_obj.mode, octave=4) # Use octave 4 for scale base
            if not scale_obj:
                 print("Error: Could not get scale object for melody generation.")
                 return None
            # Get pitches within the desired range
            scale_pitches = scale_obj.getPitches(start_pitch_range, end_pitch_range)
            if not scale_pitches:
                 print(f"Error: No scale pitches found in range {start_pitch_range}-{end_pitch_range}.")
                 return None

            total_duration_ql = measures * ts_obj.beatCount * (4 / ts_obj.denominator) # Total quarter lengths

            # --- Rhythm Generation ---
            # More nuanced rhythm generation based on complexity
            simple_durations = [0.5, 1.0, 2.0] # Eighth, Quarter, Half
            complex_durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0] # 16th to Whole + Dotted
            possible_durations = []
            weights = []

            if rhythm_complexity < 0.3:
                 possible_durations = [0.5, 1.0, 2.0, 4.0] # Mostly eighth, quarter, half, occasional whole
                 weights = [2, 4, 2, 1]
            elif rhythm_complexity < 0.6:
                 possible_durations = [0.25, 0.5, 1.0, 1.5, 2.0] # Add 16ths and dotted quarters
                 weights = [1, 4, 4, 1, 2]
            else: # High complexity
                 possible_durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] # Wider range, include dotted eighths
                 weights = [3, 4, 2, 3, 2, 1, 1]

            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0: weights = [1]*len(possible_durations); total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            rest_probability = 0.1 + (rhythm_complexity * 0.15) # More rests with higher complexity

            current_duration_ql = 0.0
            rhythm_elements = [] # Store generated notes/rests here first
            while current_duration_ql < total_duration_ql:
                 # Choose duration
                 duration_ql = random.choices(possible_durations, weights=normalized_weights, k=1)[0]

                 # Adjust duration if it exceeds remaining total time
                 if current_duration_ql + duration_ql > total_duration_ql:
                      duration_ql = total_duration_ql - current_duration_ql

                 if duration_ql <= 0: break # Avoid zero or negative duration

                 # Add note or rest
                 if random.random() < rest_probability:
                     element = music21.note.Rest()
                 else:
                     element = music21.note.Note() # Pitch assigned later

                 # Sometimes tie notes across beats/measures (more likely with complexity)
                 # Basic tie logic: if duration is long, maybe tie (simplistic)
                 # A better approach involves looking at measure position
                 # For now, music21's makeTies will handle cleanup

                 element.duration = music21.duration.Duration(quarterLength=duration_ql)
                 rhythm_elements.append(element)
                 current_duration_ql += duration_ql

            # --- Pitch Generation based on Contour ---
            num_notes_needed = sum(1 for el in rhythm_elements if isinstance(el, music21.note.Note))
            if num_notes_needed == 0:
                 # If only rests generated, add them to the part
                 for rest in rhythm_elements: melody_part.append(rest)
                 melody_part.makeMeasures(inPlace=True)
                 return melody_part # Return part with only rests

            melody_pitches = []
            # Start pitch: Tonic in the middle of the range
            start_index = min(range(len(scale_pitches)), key=lambda i: abs(scale_pitches[i].midi - tonic_pitch.transpose('P8').midi)) # Find tonic around octave 4/5
            current_pitch_index = start_index

            # Generate pitch indices based on contour
            pitch_indices = []
            for i in range(num_notes_needed):
                 target_index = current_pitch_index # Default: stay static

                 progress = i / max(1, num_notes_needed - 1) # Normalize progress 0 to 1

                 # Calculate target index based on shape
                 if contour_shape == 'ascending':
                     target_index = int(progress * (len(scale_pitches) - 1))
                 elif contour_shape == 'descending':
                     target_index = int((1 - progress) * (len(scale_pitches) - 1))
                 elif contour_shape == 'arch':
                      peak_progress = 1 - abs(progress - 0.5) * 2 # 0 -> 1 -> 0
                      target_index = int(peak_progress * (len(scale_pitches) - 1))
                 elif contour_shape == 'valley':
                      valley_progress = abs(progress - 0.5) * 2 # 1 -> 0 -> 1
                      target_index = int(valley_progress * (len(scale_pitches) - 1))
                 elif contour_shape == 'static':
                       target_index = start_index # Stay around start
                 elif contour_shape == 'random':
                      target_index = random.randint(0, len(scale_pitches) - 1)


                 # --- Movement Logic ---
                 # Determine step size towards target: stepwise, small leap, random jump
                 step = target_index - current_pitch_index
                 move = 0
                 rand_factor = random.random()

                 if abs(step) <= 1: # If close or at target
                      # Mostly stepwise, occasional same note
                      move = step if rand_factor > 0.2 else 0
                 elif abs(step) <= 3: # If moderately close
                      # Stepwise or small leap (2)
                      move = (1 if step > 0 else -1) if rand_factor < 0.7 else (2 if step > 0 else -2)
                 else: # If far from target
                      # Small leap or larger jump (3 or random within scale)
                      if rand_factor < 0.6:
                          move = (2 if step > 0 else -2)
                      elif rand_factor < 0.85:
                           move = (3 if step > 0 else -3)
                      else: # Occasional random jump within scale towards target general direction
                           move = random.randint(1, 3) * (1 if step > 0 else -1)


                 next_index = current_pitch_index + move
                 # Bound check: ensure index stays within scale_pitches list
                 next_index = max(0, min(len(scale_pitches) - 1, next_index))

                 pitch_indices.append(next_index)
                 current_pitch_index = next_index # Update current position

            # Assign generated pitches to notes
            note_idx = 0
            for element in rhythm_elements:
                if isinstance(element, music21.note.Note):
                    if note_idx < len(pitch_indices):
                         element.pitch = scale_pitches[pitch_indices[note_idx]]
                         note_idx += 1
                    else:
                         # Should not happen if num_notes_needed was calculated correctly
                         # If it does, replace with rest
                         element = music21.note.Rest(quarterLength=element.duration.quarterLength)
                # Append the (now possibly pitched) element to the part stream
                melody_part.append(element)


            # Final cleanup
            melody_part.makeMeasures(inPlace=True) # Create measures automatically
            melody_part.makeTies(inPlace=True) # Clean up ties across barlines if needed

            return melody_part

        except Exception as e:
            print(f"Error generating melody: {e}")
            print(traceback.format_exc()) # Print stack trace for debugging
            return None

# --- Example Usage (Requires music21) ---
if __name__ == "__main__":
    # This block will only run if the script is executed directly
    # AND music21 is installed in the environment.
    try:
        engine = MusicTheoryEngine()
        print("MusicTheoryEngine Initialized.")

        # Scales
        print("\n--- Scales ---")
        c_major_scale_notes = engine.get_scale_notes('C', 'major', 1)
        print(f"C Major Scale Notes: {c_major_scale_notes}")
        a_dorian_notes = engine.get_scale_notes('A', 'dorian', 2, start_octave=3)
        print(f"A Dorian (2 oct): {a_dorian_notes}")
        f_sharp_blues_notes = engine.get_scale_notes('F#', 'blues')
        print(f"F# Blues Notes: {f_sharp_blues_notes}")
        invalid_scale = engine.get_scale_notes('C', 'invalid_scale')
        print(f"Invalid Scale Test: {invalid_scale}")

        # Chords
        print("\n--- Chords ---")
        g7_chord_notes = engine.get_chord_notes('G7')
        print(f"G7 Notes (parsed from 'G7'): {g7_chord_notes}")
        cm7b5_notes = engine.get_chord_notes('C', 'm7b5', octave=3)
        print(f"Cm7b5 Notes (explicit quality, oct 3): {cm7b5_notes}")
        daug_notes = engine.get_chord_notes('Daug')
        print(f"Daug Notes (parsed from 'Daug'): {daug_notes}")
        c_notes = engine.get_chord_notes('C')
        print(f"C Notes (parsed from 'C', default major): {c_notes}")
        fsharp_min7_notes = engine.get_chord_notes('F#min7')
        print(f"F#min7 Notes (parsed from 'F#min7'): {fsharp_min7_notes}")

        # Chord Voicing
        print("\n--- Voicing ---")
        cmaj7_chord = engine.get_chord('Cmaj7')
        if cmaj7_chord:
             print(f"CMaj7 Close (default): {[p.nameWithOctave for p in sorted(cmaj7_chord.pitches)]}")
             cmaj7_drop2 = engine.apply_voicing(cmaj7_chord, 'drop2')
             print(f"CMaj7 Drop2: {[p.nameWithOctave for p in sorted(cmaj7_drop2.pitches)]}")
             cmaj7_drop3 = engine.apply_voicing(cmaj7_chord, 'drop3')
             print(f"CMaj7 Drop3: {[p.nameWithOctave for p in sorted(cmaj7_drop3.pitches)]}")
        else: print("Could not create CMaj7 chord for voicing test.")


        # Harmony ID
        print("\n--- Harmony ID ---")
        notes1 = ['C4', 'E4', 'G4']
        root1, qual1 = engine.identify_harmony(notes1)
        print(f"Harmony for {notes1}: Root='{root1}', Quality/Scale='{qual1}'") # Expect C major

        notes2 = [60, 63, 67, 70] # MIDI C4, Eb4, G4, Bb4
        root2, qual2 = engine.identify_harmony(notes2)
        print(f"Harmony for MIDI {notes2}: Root='{root2}', Quality/Scale='{qual2}'") # Expect C minor7

        notes3 = ['D4', 'F#4', 'A4', 'C5'] # D7
        root3, qual3 = engine.identify_harmony(notes3)
        print(f"Harmony for {notes3}: Root='{root3}', Quality/Scale='{qual3}'") # Expect D dominant7

        notes4 = ['C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3'] # C major scale notes
        root4, qual4 = engine.identify_harmony(notes4)
        print(f"Harmony for {notes4}: Root='{root4}', Quality/Scale='{qual4}'") # Expect C major (as scale)

        notes5 = ['G4', 'Bb4', 'Db5'] # G diminished triad
        root5, qual5 = engine.identify_harmony(notes5)
        print(f"Harmony for {notes5}: Root='{root5}', Quality/Scale='{qual5}'") # Expect G diminished

        # Harmonic Function
        print("\n--- Harmonic Function ---")
        prog_cm = ['Am', 'Dm', 'G7', 'C']
        key_c = 'C major'
        functions_c = engine.analyze_harmonic_function(prog_cm, key_c)
        print(f"Functions in {key_c} for {prog_cm}: {functions_c}") # Expect ['TD', 'PD', 'D', 'T'] or similar

        prog_am = ['Am', 'E7', 'Am', 'Bø7', 'E7', 'Am'] # ø7 is m7b5 / half-diminished
        key_am = 'A minor'
        functions_am = engine.analyze_harmonic_function(prog_am, key_am)
        print(f"Functions in {key_am} for {prog_am}: {functions_am}") # Expect ['T', 'D', 'T', 'PD' (iiø7), 'D', 'T'] or similar

        # Contour
        print("\n--- Melodic Contour ---")
        melody1 = ['C4', 'E4', 'G4', 'C5', 'G4', 'E4', 'C4'] # Arch
        print(f"Contour for {melody1}: {engine.analyze_melodic_contour(melody1)}")
        melody2 = ['G4', 'F4', 'Eb4', 'D4', 'C4', 'D4', 'F4', 'G4'] # Valley
        print(f"Contour for {melody2}: {engine.analyze_melodic_contour(melody2)}")
        melody3 = ['C4', 'C4', 'D4', 'D4', 'E4', 'F4'] # Mostly Ascending
        print(f"Contour for {melody3}: {engine.analyze_melodic_contour(melody3)}")
        melody4 = ['A4', 'G#4', 'F#4', 'G#4', 'E4'] # Undulating
        print(f"Contour for {melody4}: {engine.analyze_melodic_contour(melody4)}")

        # Progression Generation
        print("\n--- Progression Generation ---")
        p1 = engine.generate_chord_progression('G major', 'I-V7-vi-IV')
        print(f"G Major I-V7-vi-IV: {[cs.figure for cs in p1 if cs]}")
        p2 = engine.generate_chord_progression('E minor', num_chords=8) # Random generation
        print(f"E Minor Random (8 chords): {[cs.figure for cs in p2 if cs]}")
        p3 = engine.generate_chord_progression('Bb Major', 'I V/ii ii V I') # With secondary dominant
        print(f"Bb Major Secondary Dom: {[cs.figure for cs in p3 if cs]}")


        # Melody Generation
        print("\n--- Melody Generation ---")
        print("Generating C Major Melody (Arch)...")
        melody_stream = engine.generate_melody('C major', measures=4, contour_shape='arch', instrument='Piano')
        if melody_stream:
             print("Melody generated. First few elements:")
             for element in melody_stream.flat.notesAndRests[:10]: # Show first 10 notes/rests
                  if isinstance(element, music21.note.Note):
                      print(f"  Pitch: {element.pitch.nameWithOctave:<5} Duration: {element.duration.quarterLength:<4}")
                  elif isinstance(element, music21.note.Rest):
                      print(f"  Rest{'':<8} Duration: {element.duration.quarterLength:<4}")
            # To show graphically (if MusicXML viewer configured in music21 environment):
            # melody_stream.show()
            # To save as MusicXML:
            # melody_stream.write('musicxml', 'generated_melody.xml')
        else:
             print("Melody generation failed.")

    except ImportError:
         print("\n*** MUSIC21 LIBRARY NOT FOUND ***")
         print("The example usage requires the 'music21' library.")
         print("Please install it using: pip install music21")
    except Exception as main_exception:
         print(f"\n*** An error occurred during example usage: {main_exception} ***")
         print(traceback.format_exc())
