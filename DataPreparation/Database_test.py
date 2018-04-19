from Database import Database 

class TestDatabase():
    def test_chunking(self):
        db = Database("./BDD/Test/biostats.csv")
        chunks = db.sliceToChunks('Sex')
