from abc import abstractmethod

class TextCompletionModel:

    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def complete(self, prompt, args=None):
        """Should return the completion of the prompt"""
        pass
    
    @abstractmethod
    def get_num_tokens(self, prompt, args=None):
        pass

class ChatCompletionModel:

    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def complete(self, prompt, args=None):
        """Should return the completion of the prompt, appended to preexisting chat history"""
        pass
    
    @abstractmethod
    def get_num_tokens(self, prompt, args=None):
        pass

    @abstractmethod
    def clear(self):
        """Should clear the accumulated chat history"""
        pass
