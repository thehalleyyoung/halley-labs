//! Timer management with a hierarchical timer wheel.
//!
//! Provides efficient insertion and expiry of one-shot and repeating timers,
//! supporting `O(1)` start/cancel and amortised `O(1)` advance.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Opaque handle to a running timer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimerHandle(pub u64);

/// A timer expiry event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimerExpiry {
    pub handle: TimerHandle,
    pub callback_id: String,
    pub scheduled_time: f64,
    pub actual_time: f64,
}

/// Internal timer entry.
#[derive(Debug, Clone)]
struct TimerEntry {
    handle: TimerHandle,
    expiry_time: f64,
    callback_id: String,
    repeating: bool,
    interval: f64,
}

// ---------------------------------------------------------------------------
// Timer wheel
// ---------------------------------------------------------------------------

/// Number of slots in the wheel.
const WHEEL_SIZE: usize = 256;

/// A single slot in the timer wheel — just a list of timers.
#[derive(Debug, Clone, Default)]
struct WheelSlot {
    entries: Vec<TimerEntry>,
}

// ---------------------------------------------------------------------------
// TimerManager
// ---------------------------------------------------------------------------

/// Manages timers using a hashed timer wheel.
#[derive(Debug)]
pub struct TimerManager {
    /// Current simulation time.
    current_time: f64,
    /// Resolution of each slot (seconds).
    tick_duration: f64,
    /// Wheel slots.
    wheel: Vec<WheelSlot>,
    /// Overflow list for timers beyond one wheel revolution.
    overflow: Vec<TimerEntry>,
    /// Lookup from handle to slot index (for cancellation).
    handle_to_slot: HashMap<TimerHandle, usize>,
    /// Cancelled handles (lazy deletion).
    cancelled: HashMap<TimerHandle, bool>,
    /// Next handle id.
    next_handle: u64,
    /// Total number of active (non-cancelled) timers.
    active_count: usize,
}

impl TimerManager {
    /// Create a new timer manager.
    ///
    /// `tick_duration` is the time resolution per wheel slot (e.g. 0.01 for 10 ms).
    pub fn new(tick_duration: f64) -> Self {
        Self {
            current_time: 0.0,
            tick_duration: if tick_duration > 0.0 {
                tick_duration
            } else {
                0.01
            },
            wheel: (0..WHEEL_SIZE).map(|_| WheelSlot::default()).collect(),
            overflow: Vec::new(),
            handle_to_slot: HashMap::new(),
            cancelled: HashMap::new(),
            next_handle: 1,
            active_count: 0,
        }
    }

    /// Start a one-shot timer that expires after `duration` seconds.
    pub fn start_timer(&mut self, duration: f64, callback_id: impl Into<String>) -> TimerHandle {
        let handle = TimerHandle(self.next_handle);
        self.next_handle += 1;

        let entry = TimerEntry {
            handle,
            expiry_time: self.current_time + duration,
            callback_id: callback_id.into(),
            repeating: false,
            interval: duration,
        };

        self.insert_entry(entry);
        self.active_count += 1;
        handle
    }

    /// Start a repeating timer that fires every `interval` seconds.
    pub fn start_repeating(
        &mut self,
        interval: f64,
        callback_id: impl Into<String>,
    ) -> TimerHandle {
        let handle = TimerHandle(self.next_handle);
        self.next_handle += 1;

        let entry = TimerEntry {
            handle,
            expiry_time: self.current_time + interval,
            callback_id: callback_id.into(),
            repeating: true,
            interval,
        };

        self.insert_entry(entry);
        self.active_count += 1;
        handle
    }

    /// Cancel a timer. Returns true if the timer was found and cancelled.
    pub fn cancel_timer(&mut self, handle: TimerHandle) -> bool {
        if self.cancelled.contains_key(&handle) {
            return false;
        }
        // Mark as cancelled (lazy deletion)
        self.cancelled.insert(handle, true);
        self.active_count = self.active_count.saturating_sub(1);
        true
    }

    /// Advance time by `delta` seconds and return all expired timers.
    pub fn advance_time(&mut self, delta: f64) -> Vec<TimerExpiry> {
        let target_time = self.current_time + delta;
        let mut expired = Vec::new();

        while self.current_time < target_time {
            let step = self.tick_duration.min(target_time - self.current_time);
            self.current_time += step;

            let slot_idx = self.time_to_slot(self.current_time);
            let slot = &mut self.wheel[slot_idx];

            // Drain entries from the current slot
            let mut remaining = Vec::new();
            let entries = std::mem::take(&mut slot.entries);
            for entry in entries {
                if self.cancelled.remove(&entry.handle).is_some() {
                    continue; // cancelled
                }
                if entry.expiry_time <= self.current_time + self.tick_duration * 0.5 {
                    expired.push(TimerExpiry {
                        handle: entry.handle,
                        callback_id: entry.callback_id.clone(),
                        scheduled_time: entry.expiry_time,
                        actual_time: self.current_time,
                    });
                    if entry.repeating {
                        // Re-insert repeating timer
                        let new_entry = TimerEntry {
                            handle: entry.handle,
                            expiry_time: entry.expiry_time + entry.interval,
                            callback_id: entry.callback_id,
                            repeating: true,
                            interval: entry.interval,
                        };
                        self.insert_entry_no_count(new_entry);
                    } else {
                        self.active_count = self.active_count.saturating_sub(1);
                    }
                } else {
                    remaining.push(entry);
                }
            }
            self.wheel[slot_idx].entries = remaining;

            // Check overflow for entries that should now be in the wheel
            self.migrate_overflow();
        }

        self.current_time = target_time;
        expired
    }

    /// Get the current time.
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Number of active timers.
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Check if a timer is active.
    pub fn is_active(&self, handle: TimerHandle) -> bool {
        !self.cancelled.contains_key(&handle) && self.handle_to_slot.contains_key(&handle)
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn time_to_slot(&self, time: f64) -> usize {
        let ticks = (time / self.tick_duration) as usize;
        ticks % WHEEL_SIZE
    }

    fn insert_entry(&mut self, entry: TimerEntry) {
        self.insert_entry_no_count(entry);
    }

    fn insert_entry_no_count(&mut self, entry: TimerEntry) {
        let ticks_until = ((entry.expiry_time - self.current_time) / self.tick_duration).ceil() as usize;
        if ticks_until < WHEEL_SIZE {
            let slot_idx = self.time_to_slot(entry.expiry_time);
            self.handle_to_slot.insert(entry.handle, slot_idx);
            self.wheel[slot_idx].entries.push(entry);
        } else {
            self.handle_to_slot
                .insert(entry.handle, usize::MAX);
            self.overflow.push(entry);
        }
    }

    fn migrate_overflow(&mut self) {
        let mut remaining = Vec::new();
        let overflow = std::mem::take(&mut self.overflow);
        for entry in overflow {
            if self.cancelled.contains_key(&entry.handle) {
                continue;
            }
            let ticks_until =
                ((entry.expiry_time - self.current_time) / self.tick_duration).ceil() as usize;
            if ticks_until < WHEEL_SIZE {
                let slot_idx = self.time_to_slot(entry.expiry_time);
                self.handle_to_slot.insert(entry.handle, slot_idx);
                self.wheel[slot_idx].entries.push(entry);
            } else {
                remaining.push(entry);
            }
        }
        self.overflow = remaining;
    }
}

impl Default for TimerManager {
    fn default() -> Self {
        Self::new(0.01)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_shot_timer() {
        let mut tm = TimerManager::new(0.01);
        let _h = tm.start_timer(1.0, "cb1");
        assert_eq!(tm.active_count(), 1);

        let expired = tm.advance_time(0.5);
        assert!(expired.is_empty());

        let expired = tm.advance_time(0.6);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].callback_id, "cb1");
    }

    #[test]
    fn timer_cancel() {
        let mut tm = TimerManager::new(0.01);
        let h = tm.start_timer(1.0, "cb1");
        assert!(tm.cancel_timer(h));
        assert_eq!(tm.active_count(), 0);

        let expired = tm.advance_time(2.0);
        assert!(expired.is_empty());
    }

    #[test]
    fn repeating_timer() {
        let mut tm = TimerManager::new(0.01);
        let _h = tm.start_repeating(0.5, "repeat_cb");
        let mut total_fires = 0;

        for _ in 0..10 {
            let expired = tm.advance_time(0.5);
            total_fires += expired.len();
        }
        // Should fire approximately 10 times over 5 seconds with 0.5s interval
        assert!(total_fires >= 8, "Expected ~10 fires, got {}", total_fires);
    }

    #[test]
    fn multiple_timers_different_durations() {
        let mut tm = TimerManager::new(0.01);
        let _h1 = tm.start_timer(0.5, "short");
        let _h2 = tm.start_timer(1.5, "long");

        let exp1 = tm.advance_time(0.6);
        assert_eq!(exp1.len(), 1);
        assert_eq!(exp1[0].callback_id, "short");

        let exp2 = tm.advance_time(1.0);
        assert_eq!(exp2.len(), 1);
        assert_eq!(exp2[0].callback_id, "long");
    }

    #[test]
    fn current_time_advances() {
        let mut tm = TimerManager::new(0.01);
        assert_eq!(tm.current_time(), 0.0);
        tm.advance_time(3.14);
        assert!((tm.current_time() - 3.14).abs() < 0.001);
    }

    #[test]
    fn cancel_returns_false_for_unknown() {
        let mut tm = TimerManager::new(0.01);
        assert!(!tm.cancel_timer(TimerHandle(999)));
    }

    #[test]
    fn double_cancel() {
        let mut tm = TimerManager::new(0.01);
        let h = tm.start_timer(1.0, "cb");
        assert!(tm.cancel_timer(h));
        assert!(!tm.cancel_timer(h));
    }

    #[test]
    fn zero_duration_timer() {
        let mut tm = TimerManager::new(0.01);
        let _h = tm.start_timer(0.0, "instant");
        let expired = tm.advance_time(0.01);
        assert_eq!(expired.len(), 1);
    }

    #[test]
    fn overflow_timer() {
        // Timer longer than one full wheel revolution
        let mut tm = TimerManager::new(0.01);
        let _h = tm.start_timer(10.0, "overflow");
        let expired = tm.advance_time(9.0);
        assert!(expired.is_empty());
        let expired = tm.advance_time(1.1);
        assert_eq!(expired.len(), 1);
    }

    #[test]
    fn many_timers() {
        let mut tm = TimerManager::new(0.01);
        for i in 0..100 {
            tm.start_timer(i as f64 * 0.01 + 0.01, &format!("cb{}", i));
        }
        assert_eq!(tm.active_count(), 100);
        let expired = tm.advance_time(2.0);
        assert_eq!(expired.len(), 100);
    }
}
