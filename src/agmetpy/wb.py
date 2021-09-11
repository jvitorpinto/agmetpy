
class Event:
    def __init__(self, parent):
        self._parent = parent
        self._subscriptions = []
    
    def subscribe(self, event_handler):
        self._subscriptions.append(event_handler)
    
    def unsubscribe(self, event_handler):
        self._subscriptions.remove(event_handler)
    
    def dispatch(self, event_args):
        for event_handler in self._subscriptions:
            event_handler(self._parent, event_args)

class SimulationObject:
    '''
    Represents an object that is initialized and then updated
    at each step of a simulation.
    '''
    def __init__(self):
        self._simulation = None
    
    def initialize(self):
        pass
    
    def update(self):
        pass
    

