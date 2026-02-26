"""Unit tests for soundness level tracking."""

import pytest

from bioprover.soundness import SoundnessAnnotation, SoundnessLevel


class TestSoundnessLevel:
    def test_ordering(self):
        assert SoundnessLevel.SOUND < SoundnessLevel.DELTA_SOUND
        assert SoundnessLevel.DELTA_SOUND < SoundnessLevel.BOUNDED
        assert SoundnessLevel.BOUNDED < SoundnessLevel.APPROXIMATE

    def test_le(self):
        assert SoundnessLevel.SOUND <= SoundnessLevel.SOUND
        assert SoundnessLevel.SOUND <= SoundnessLevel.APPROXIMATE

    def test_ge(self):
        assert SoundnessLevel.APPROXIMATE >= SoundnessLevel.SOUND
        assert SoundnessLevel.SOUND >= SoundnessLevel.SOUND

    def test_not_less_than_self(self):
        assert not (SoundnessLevel.SOUND < SoundnessLevel.SOUND)

    def test_meet_same(self):
        assert SoundnessLevel.meet(SoundnessLevel.SOUND, SoundnessLevel.SOUND) == SoundnessLevel.SOUND

    def test_meet_different(self):
        result = SoundnessLevel.meet(SoundnessLevel.SOUND, SoundnessLevel.DELTA_SOUND)
        assert result == SoundnessLevel.DELTA_SOUND

    def test_meet_commutative(self):
        a = SoundnessLevel.meet(SoundnessLevel.BOUNDED, SoundnessLevel.DELTA_SOUND)
        b = SoundnessLevel.meet(SoundnessLevel.DELTA_SOUND, SoundnessLevel.BOUNDED)
        assert a == b == SoundnessLevel.BOUNDED

    def test_meet_returns_weakest(self):
        result = SoundnessLevel.meet(SoundnessLevel.SOUND, SoundnessLevel.APPROXIMATE)
        assert result == SoundnessLevel.APPROXIMATE


class TestSoundnessAnnotation:
    def test_initial_sound(self):
        ann = SoundnessAnnotation(level=SoundnessLevel.SOUND)
        assert ann.level == SoundnessLevel.SOUND
        assert ann.assumptions == []

    def test_weaken_to(self):
        ann = SoundnessAnnotation(level=SoundnessLevel.SOUND)
        weakened = ann.weaken_to(SoundnessLevel.BOUNDED, "bounded model checking")
        assert weakened.level == SoundnessLevel.BOUNDED
        assert "bounded model checking" in weakened.assumptions

    def test_weaken_preserves_existing_assumptions(self):
        ann = SoundnessAnnotation(level=SoundnessLevel.SOUND, assumptions=["initial"])
        weakened = ann.weaken_to(SoundnessLevel.BOUNDED, "reason")
        assert len(weakened.assumptions) == 2
        assert weakened.assumptions[0] == "initial"

    def test_weaken_does_not_strengthen(self):
        ann = SoundnessAnnotation(level=SoundnessLevel.APPROXIMATE)
        weakened = ann.weaken_to(SoundnessLevel.SOUND, "attempting to strengthen")
        assert weakened.level == SoundnessLevel.APPROXIMATE

    def test_with_delta(self):
        ann = SoundnessAnnotation(level=SoundnessLevel.SOUND)
        delta_ann = ann.with_delta(1e-3)
        assert delta_ann.level == SoundnessLevel.DELTA_SOUND
        assert delta_ann.delta == 1e-3
        assert any("delta" in a for a in delta_ann.assumptions)

    def test_with_time_bound(self):
        ann = SoundnessAnnotation(level=SoundnessLevel.SOUND)
        bounded = ann.with_time_bound(10.0)
        assert bounded.level == SoundnessLevel.BOUNDED
        assert bounded.time_bound == 10.0

    def test_chained_weakening(self):
        ann = SoundnessAnnotation(level=SoundnessLevel.SOUND)
        ann = ann.with_delta(1e-3)
        ann = ann.with_time_bound(10.0)
        # BOUNDED is weaker than DELTA_SOUND, so level should be BOUNDED
        assert ann.level == SoundnessLevel.BOUNDED
        assert len(ann.assumptions) == 2
