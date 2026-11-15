from app.analytics import LineCounter, ZoneTracker, point_in_polygon, segments_intersect


def test_segments_intersect():
    assert segments_intersect((0.5, 0.4), (0.5, 0.6), (0, 0.5), (1, 0.5))
    assert not segments_intersect((0.5, 0.1), (0.5, 0.3), (0, 0.5), (1, 0.5))


def test_line_counts_in_and_out():
    lc = LineCounter("door", (0.0, 0.5), (1.0, 0.5))
    # walking down (towards larger y) = "in" by default
    assert lc.update(1, (0.5, 0.40), 0.0) is None
    c = lc.update(1, (0.5, 0.60), 0.5)
    assert c and c.direction == "in"
    # another person walking up = "out"
    lc.update(2, (0.3, 0.70), 0.0)
    c = lc.update(2, (0.3, 0.30), 0.4)
    assert c and c.direction == "out"
    assert (lc.count_in, lc.count_out) == (1, 1)


def test_invert_flips_direction():
    lc = LineCounter("door", (0.0, 0.5), (1.0, 0.5), invert=True)
    lc.update(1, (0.5, 0.4), 0)
    assert lc.update(1, (0.5, 0.6), 1).direction == "out"


def test_crossing_outside_segment_is_ignored():
    lc = LineCounter("door", (0.4, 0.5), (0.6, 0.5))
    lc.update(1, (0.9, 0.4), 0)
    assert lc.update(1, (0.9, 0.6), 1) is None


def test_jitter_cooldown():
    lc = LineCounter("door", (0.0, 0.5), (1.0, 0.5), cooldown_s=1.5)
    lc.update(1, (0.5, 0.49), 0.0)
    assert lc.update(1, (0.5, 0.51), 0.1)
    assert lc.update(1, (0.5, 0.49), 0.2) is None  # bounced back on the line
    assert lc.count_in == 1 and lc.count_out == 0


def test_point_in_polygon():
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert point_in_polygon((0.5, 0.5), sq)
    assert not point_in_polygon((1.5, 0.5), sq)


def test_zone_dwell():
    z = ZoneTracker("reception", [(0, 0), (1, 0), (1, 1), (0, 1)], exit_grace_s=2, min_dwell_s=1)
    for t in range(0, 11):
        z.update(7, (0.5, 0.5), float(t))
    assert z.occupancy == 1
    assert z.flush(11.0) == []          # still within grace period
    visits = z.flush(13.5)
    assert len(visits) == 1 and visits[0].dwell_s == 10
    assert z.occupancy == 0


def test_zone_ignores_pass_through():
    z = ZoneTracker("corner", [(0, 0), (1, 0), (1, 1), (0, 1)], min_dwell_s=3)
    z.update(1, (0.5, 0.5), 0.0)
    z.update(1, (0.5, 0.5), 1.0)
    assert z.flush(10.0) == []
