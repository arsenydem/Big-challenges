class UserStateManager:
    def __init__(self):
        self.states = {}  

    def set_state(self, user_id, state, age_range=None):
        self.states[user_id] = {'state': state, 'range': age_range}

    def get_state(self, user_id):
        return self.states.get(user_id, {'state': None, 'range': None})

    def clear_state(self, user_id):
        if user_id in self.states:
            del self.states[user_id]