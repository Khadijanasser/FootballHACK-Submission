"""Temporary invented FIFA/EPTS and SportsCode files, never competition assets."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest


@pytest.fixture
def atd_factory(tmp_path):
    def build(*, n=40, fps=25, pitch=(100, 60), normalized=True,
              periods=None, players=None, ball=None, record_frames=None,
              specs=True, orders=None, namespace=False, trailing=False, folder="fixture"):
        directory = tmp_path / folder
        directory.mkdir(parents=True, exist_ok=True)
        metadata, raw = directory / "invented_atd.xml", directory / "invented_atd.txt"
        # Non-numeric/non-sorted IDs and interleaved membership are intentional.
        slot_ids = ("slot-9", "slot-2", "slot-1")
        side_ids = ("home-id", "away-id", "home-id")
        if ball is None:
            ball = np.column_stack((.5 + np.arange(n) * .002, np.full(n, .5)))
        ball = np.asarray(ball, dtype=float)
        if players is None:
            players = np.stack((ball, ball + [.1, 0], ball + [.2, 0]), axis=1)
        players = np.asarray(players, dtype=float)
        if not normalized:
            ball, players = ball * pitch, players * pitch
        root = ET.Element("FIFADataTransfer", {"xmlns": "urn:invented:epts"} if namespace else {})
        meta = ET.SubElement(root, "Metadata")
        global_config = ET.SubElement(meta, "GlobalConfig")
        ET.SubElement(global_config, "ProviderName").text = "Metrica Sports"
        ET.SubElement(global_config, "FrameRate").text = str(fps)
        provider = ET.SubElement(global_config, "ProviderGlobalParameters")
        if periods is not None:
            for i, (start, end) in enumerate(periods):
                for suffix, value in (("start", start + 1), ("end", end)):
                    param = ET.SubElement(provider, "ProviderParameter")
                    ET.SubElement(param, "Name").text = f'{("first_half", "second_half")[i]}_{suffix}'
                    ET.SubElement(param, "Value").text = str(value)
        session = ET.SubElement(ET.SubElement(meta, "Sessions"), "Session")
        field = ET.SubElement(session, "FieldSize")
        ET.SubElement(field, "Width").text = str(pitch[0])
        ET.SubElement(field, "Height").text = str(pitch[1])
        teams = ET.SubElement(meta, "Teams")
        for team_id, name in (("home-id", "Synthetic Home"), ("away-id", "Synthetic Away")):
            team = ET.SubElement(teams, "Team", id=team_id)
            ET.SubElement(team, "Name").text = name
        player_nodes = ET.SubElement(meta, "Players")
        for slot_id, team_id in zip(slot_ids, side_ids):
            node = ET.SubElement(player_nodes, "Player", id=slot_id, teamId=team_id)
            ET.SubElement(node, "Name").text = "Track " + slot_id
            ET.SubElement(node, "ShirtNumber").text = "999"  # never an identity/key
        sensors = ET.SubElement(ET.SubElement(ET.SubElement(meta, "Devices"), "Device"), "Sensors")
        sensor = ET.SubElement(sensors, "Sensor", id="position")
        channels = ET.SubElement(sensor, "Channels")
        for channel in ("x", "y", "s"):
            node = ET.SubElement(channels, "Channel", id=channel)
            ET.SubElement(node, "Name").text = channel
            ET.SubElement(node, "Unit").text = ("normalized" if normalized else "m") if channel != "s" else "m/s"
        refs = ET.SubElement(meta, "PlayerChannels")
        for slot_id in slot_ids:
            for channel in ("x", "y", "s"):
                ET.SubElement(refs, "PlayerChannel", id=f"{slot_id}-{channel}",
                              playerId=slot_id, channelId=channel)
        orders = [(1, n, (0, 1, 2))] if orders is None else orders
        if specs:
            layouts = ET.SubElement(root, "DataFormatSpecifications")
            for first, last, order in orders:
                spec = ET.SubElement(layouts, "DataFormatSpecification", startFrame=str(first), endFrame=str(last), separator=":")
                ET.SubElement(spec, "StringRegister", name="frameCount")
                player_register = ET.SubElement(spec, "SplitRegister", separator=";")
                for slot in order:
                    triple = ET.SubElement(player_register, "SplitRegister", separator=",")
                    for channel in ("x", "y", "s"):
                        ET.SubElement(triple, "PlayerChannelRef", playerChannelId=f"{slot_ids[slot]}-{channel}")
                triple = ET.SubElement(spec, "SplitRegister", separator=",")
                for channel in ("x", "y", "s"):
                    ET.SubElement(triple, "BallChannelRef", channelId=channel)
        ET.ElementTree(root).write(metadata, encoding="utf-8", xml_declaration=True)

        def triple(xy):
            return ",".join("NaN" if np.isnan(value) else format(value, ".15g") for value in (*xy, 2.0)) + ("," if trailing else "")

        with raw.open("w", encoding="utf-8", newline="") as handle:
            for frame in range(n) if record_frames is None else record_frames:
                order = next(order for first, last, order in orders if first <= frame + 1 <= last)
                player_text = ";".join(triple(players[frame, slot]) for slot in order)
                if trailing:
                    player_text += ";"
                line = f"{frame + 1}:{player_text}:{triple(ball[frame])}" + (":" if trailing else "")
                handle.write(line + "\n")
        return metadata, raw, players, ball
    return build


@pytest.fixture
def tagging_factory(tmp_path):
    def build(events, *, fps=25, namespace=False, name="invented_tagging.xml"):
        root = ET.Element("file", {"xmlns": "urn:invented:sportscode"} if namespace else {})
        instances = ET.SubElement(root, "ALL_INSTANCES")
        for i, event in enumerate(events, 1):
            node = ET.SubElement(instances, "instance")
            ET.SubElement(node, "ID").text = str(i)
            for key in ("start", "end"):
                seconds = event.get(key + "_s", event.get(key, 0) / fps)
                ET.SubElement(node, key).text = str(seconds)
            ET.SubElement(node, "code").text = event.get("code", "")
            for group, text in event.get("labels", []):
                label = ET.SubElement(node, "label")
                ET.SubElement(label, "group").text = group
                ET.SubElement(label, "text").text = text
        path = tmp_path / name
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        return path
    return build
