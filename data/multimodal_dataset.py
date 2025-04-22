
class MultimodalDataset:
    def __init__(self, vision_data, text_data, audio_data, batch_size=8):
        self.vision = vision_data
        self.text = text_data
        self.audio = audio_data
        self.batch_size = batch_size
        self.index = 0
        self.total = len(vision_data)

    def __len__(self):
        return self.total // self.batch_size

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.total:
            raise StopIteration
        start, end = self.index, self.index + self.batch_size
        self.index += self.batch_size
        return {
            'vision': self.vision[start:end],
            'text': self.text[start:end],
            'audio': self.audio[start:end]
        }
