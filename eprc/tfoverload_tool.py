#!/usr/bin/env python3
#
#
#
import uuid

class TFOverload_Tool:
    def __init__(self):
        self.uuid = uuid.uuid4()

    def whoami(self):
        print(self.uuid)

#
# EOF
