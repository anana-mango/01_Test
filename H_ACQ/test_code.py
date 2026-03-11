from HEAD import Dispatch
eBlockTypeVoIP = 18
 
 lc = Dispatch("MfeControlLib.labCoreControl")
 if (lc.Count != 0):
    labC = lc.Items(0)
    lc_voip = labC.AudioWiring.Blocks.FirstBlockByType(eBlockTypeVoIP)
    if lc_voip is not None:
        voip_if = lc_voip.Settings
