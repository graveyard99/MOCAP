# Contacts

The pipeline estimates left heel, left forefoot, right heel and right forefoot contact states from calibrated floor distance, motion, confidence and temporal hysteresis. Low velocity alone is insufficient to declare a planted foot.

The bottom contact timeline contains four foot tracks plus Missing data. Green marks planted samples, amber possible contact, and red missing joint evidence. Click a track to seek world time. A blank track can legitimately mean no reliable contact.

To provide a manual contact interval, set the timeline Range start/end in seconds. Open Quality Control, choose the foot and click Mark range as contact. The interval is stored as an audited override. Rerun Refine contacts… to consume the correction and update the character.

Contact refinement reduces planted horizontal motion and floor penetration subject to observation-confidence bounds. It must not rewrite strongly measured global motion to make a statistical walk cycle look better. Swing phases remain free.

Review geometric sliding, fitted sliding before refinement and fitted sliding after refinement in the JSON/HTML report. The synthetic demo may have no reliable planted interval for a particular short range; do not fabricate one to fill the timeline.

Heel/toe model joints are mapped explicitly to available ankle/foot landmarks. That mapping is reported, so sparse foot observations are not mistaken for directly measured heel markers.
