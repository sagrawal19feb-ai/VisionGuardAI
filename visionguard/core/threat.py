"""
Threat Assessment Module
========================
Combines face recognition and object detection
into a single threat level.
"""

import logging

logger = logging.getLogger(__name__)


class ThreatAssessment:
    def __init__(self, config):
        self.config = config

    def assess(self, face_results, obj_detections):
        person_status = self._person_status(face_results)

        object_threat = self._object_threat(obj_detections)

        threat_level = self._compute_threat(person_status, object_threat)

        known_persons = [f for f in face_results if getattr(f, "is_known", False)]

        unknown_persons = [f for f in face_results if not getattr(f, "is_known", False)]

        hazardous_objects = [
            o for o in obj_detections if getattr(o, "is_hazardous", False)
        ]

        return {
            "threat_level": threat_level,
            "person_status": person_status,
            "object_threat": object_threat,
            "known_persons": [p.name for p in known_persons],
            "unknown_count": len(unknown_persons),
            "hazardous_objects": [(o.label, o.confidence) for o in hazardous_objects],
            "all_objects": [(o.label, o.confidence) for o in obj_detections],
            "hex_color": self.config.THREAT_LEVELS[threat_level]["hex"],
            "bgr_color": self.config.THREAT_LEVELS[threat_level]["bgr"],
            "priority": self.config.THREAT_LEVELS[threat_level]["priority"],
            "description": self._description(
                known_persons, unknown_persons, hazardous_objects
            ),
        }

    def _person_status(self, faces):
        if not faces:
            return "none"

        for face in faces:
            if not face.is_known:
                return "unknown"

        return "known"

    def _object_threat(self, detections):
        priority = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

        highest = None
        score = -1

        for obj in detections:
            if not obj.is_hazardous:
                continue

            current = priority.get(obj.threat_modifier, 0)

            if current > score:
                score = current

                highest = obj.threat_modifier

        return highest

    def _compute_threat(self, person_status, object_threat):
        key = (person_status, object_threat)

        if key in self.config.THREAT_RULES:
            return self.config.THREAT_RULES[key]

        if person_status == "unknown":
            if object_threat in ("HIGH", "CRITICAL"):
                return "CRITICAL"

            return "HIGH"

        if object_threat == "CRITICAL":
            return "CRITICAL"

        if object_threat == "HIGH":
            return "HIGH"

        if object_threat == "MEDIUM":
            return "MEDIUM"

        return "LOW"

    def _description(self, known, unknown, hazardous):
        parts = []

        if known:
            parts.append("Known: " + ", ".join(p.name for p in known))

        if unknown:
            parts.append(f"{len(unknown)} Unknown Person(s)")

        if hazardous:
            parts.append("Hazards: " + ", ".join(o.label for o in hazardous))

        if not parts:
            return "No threats detected"

        return " | ".join(parts)
