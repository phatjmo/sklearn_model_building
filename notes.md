HUMAN = 1 --> No IVR
MACHINE:GY = 2 --> IVR with invalid DTMF Response --> Put on Graylist
MACHINE:GN = 3 --> IVR with no DTMF --> Put on Graylist
MACHINE:WY = 4 --> IVR with correct DTMF Response --> Put on Whitelist
BLANK/Otherwise = 0 --> No result/invalid result from IVR