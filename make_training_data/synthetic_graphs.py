class FakeNode():
    def __init__(self, id):
        self.id = id
        self.properties = {"timestamp": id}
        self.labels = {}


class FakeEdge():
    def __init__(self, id, start, end):
        self.id = id
        self.start = start
        self.end = end
        self.properties = {}
        self.type = ""
