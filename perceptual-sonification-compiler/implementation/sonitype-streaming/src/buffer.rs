//! Lock-free buffer infrastructure for real-time audio streaming.
//!
//! Provides single-producer single-consumer ring buffers, triple buffering,
//! specialized audio ring buffers, and lock-free event queues.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// RingBuffer<T> — SPSC lock-free ring buffer
// ---------------------------------------------------------------------------

/// Single-producer single-consumer lock-free ring buffer.
///
/// Uses power-of-two sizing with atomic read/write indices so that producer
/// and consumer can operate concurrently without locks.
pub struct RingBuffer<T> {
    storage: Box<[Option<T>]>,
    capacity: usize,
    mask: usize,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
}

unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer whose capacity is rounded up to the next
    /// power of two (minimum 2).
    pub fn new(requested: usize) -> Self {
        let capacity = requested.next_power_of_two().max(2);
        let mut v = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            v.push(None);
        }
        Self {
            storage: v.into_boxed_slice(),
            capacity,
            mask: capacity - 1,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn available_read(&self) -> usize {
        let w = self.write_pos.load(Ordering::Acquire);
        let r = self.read_pos.load(Ordering::Acquire);
        w.wrapping_sub(r)
    }

    pub fn available_write(&self) -> usize {
        self.capacity - self.available_read()
    }

    pub fn is_empty(&self) -> bool {
        self.available_read() == 0
    }

    pub fn is_full(&self) -> bool {
        self.available_write() == 0
    }

    /// Try to push a value. Returns `Err(value)` if full.
    pub fn try_push(&self, value: T) -> Result<(), T> {
        let w = self.write_pos.load(Ordering::Relaxed);
        let r = self.read_pos.load(Ordering::Acquire);
        if w.wrapping_sub(r) >= self.capacity {
            return Err(value);
        }
        let idx = w & self.mask;
        // SAFETY: single producer — only one writer at a time.
        let slot = unsafe { &mut *(self.storage.as_ptr().add(idx) as *mut Option<T>) };
        *slot = Some(value);
        self.write_pos.store(w.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Push a value, returning `false` if the buffer was full.
    pub fn push(&self, value: T) -> bool {
        self.try_push(value).is_ok()
    }

    /// Try to pop a value. Returns `None` if empty.
    pub fn try_pop(&self) -> Option<T> {
        let r = self.read_pos.load(Ordering::Relaxed);
        let w = self.write_pos.load(Ordering::Acquire);
        if r == w {
            return None;
        }
        let idx = r & self.mask;
        let slot = unsafe { &mut *(self.storage.as_ptr().add(idx) as *mut Option<T>) };
        let value = slot.take();
        self.read_pos.store(r.wrapping_add(1), Ordering::Release);
        value
    }

    /// Pop a value. Returns `None` when empty.
    pub fn pop(&self) -> Option<T> {
        self.try_pop()
    }

    /// Discard all items (consumer side).
    pub fn clear(&self) {
        while self.try_pop().is_some() {}
    }
}

// ---------------------------------------------------------------------------
// TripleBuffer<T> — writer/reader with zero-wait reads
// ---------------------------------------------------------------------------

/// Triple buffer providing zero-wait reads for a single writer / single reader.
///
/// The writer writes into the *back* buffer, then swaps it to the *middle*.
/// The reader always gets the latest completed *middle* buffer.
pub struct TripleBuffer<T: Clone> {
    buffers: [std::cell::UnsafeCell<T>; 3],
    /// Index of the middle buffer (shared between writer and reader).
    middle: AtomicUsize,
    /// Flag indicating the middle buffer has fresh data for the reader.
    fresh: AtomicBool,
    /// Writer's current back-buffer index.
    writer_idx: AtomicUsize,
    /// Reader's current front-buffer index.
    reader_idx: AtomicUsize,
}

unsafe impl<T: Clone + Send> Send for TripleBuffer<T> {}
unsafe impl<T: Clone + Send> Sync for TripleBuffer<T> {}

impl<T: Clone> TripleBuffer<T> {
    pub fn new(initial: T) -> Self {
        Self {
            buffers: [
                std::cell::UnsafeCell::new(initial.clone()),
                std::cell::UnsafeCell::new(initial.clone()),
                std::cell::UnsafeCell::new(initial),
            ],
            middle: AtomicUsize::new(1),
            fresh: AtomicBool::new(false),
            writer_idx: AtomicUsize::new(0),
            reader_idx: AtomicUsize::new(2),
        }
    }

    /// Get a mutable reference to the writer's back buffer.
    pub fn write_buf(&self) -> &mut T {
        let idx = self.writer_idx.load(Ordering::Relaxed);
        unsafe { &mut *self.buffers[idx].get() }
    }

    /// Publish the back buffer: swap it into the middle slot.
    pub fn publish(&self) {
        let old_writer = self.writer_idx.load(Ordering::Relaxed);
        let old_middle = self.middle.swap(old_writer, Ordering::AcqRel);
        self.writer_idx.store(old_middle, Ordering::Relaxed);
        self.fresh.store(true, Ordering::Release);
    }

    /// Read the latest published data. Returns `true` if new data was
    /// available since the last read.
    pub fn read(&self) -> (&T, bool) {
        let was_fresh = self.fresh.load(Ordering::Acquire);
        if was_fresh {
            let old_reader = self.reader_idx.load(Ordering::Relaxed);
            let old_middle = self.middle.swap(old_reader, Ordering::AcqRel);
            self.reader_idx.store(old_middle, Ordering::Relaxed);
            self.fresh.store(false, Ordering::Release);
        }
        let idx = self.reader_idx.load(Ordering::Relaxed);
        (unsafe { &*self.buffers[idx].get() }, was_fresh)
    }

    /// Check whether fresh data is available without consuming it.
    pub fn has_fresh_data(&self) -> bool {
        self.fresh.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// AudioRingBuffer — specialised f32 ring buffer for audio
// ---------------------------------------------------------------------------

/// Specialised ring buffer for `f32` audio samples with chunk I/O and metering.
pub struct AudioRingBuffer {
    inner: RingBuffer<f32>,
    channels: usize,
    peak_level: AtomicU64,
    rms_accumulator: AtomicU64,
    rms_count: AtomicUsize,
}

impl AudioRingBuffer {
    pub fn new(capacity: usize, channels: usize) -> Self {
        Self {
            inner: RingBuffer::new(capacity),
            channels: channels.max(1),
            peak_level: AtomicU64::new(0),
            rms_accumulator: AtomicU64::new(0),
            rms_count: AtomicUsize::new(0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn available_read(&self) -> usize {
        self.inner.available_read()
    }

    pub fn available_write(&self) -> usize {
        self.inner.available_write()
    }

    /// Write a chunk of interleaved samples.
    pub fn write_interleaved(&self, samples: &[f32]) -> usize {
        let mut written = 0;
        for &s in samples {
            if self.inner.push(s) {
                self.update_meter(s);
                written += 1;
            } else {
                break;
            }
        }
        written
    }

    /// Read a chunk of interleaved samples into `dest`. Returns how many
    /// samples were read.
    pub fn read_interleaved(&self, dest: &mut [f32]) -> usize {
        let mut count = 0;
        for slot in dest.iter_mut() {
            if let Some(s) = self.inner.pop() {
                *slot = s;
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Write non-interleaved (per-channel) buffers. `channel_data[ch]` holds
    /// samples for channel `ch`. Interleaves internally.
    pub fn write_non_interleaved(&self, channel_data: &[&[f32]]) -> usize {
        if channel_data.is_empty() {
            return 0;
        }
        let frames = channel_data.iter().map(|c| c.len()).min().unwrap_or(0);
        let mut written = 0;
        for f in 0..frames {
            let mut frame_ok = true;
            for ch in channel_data.iter() {
                if !self.inner.push(ch[f]) {
                    frame_ok = false;
                    break;
                }
                self.update_meter(ch[f]);
            }
            if frame_ok {
                written += 1;
            } else {
                break;
            }
        }
        written
    }

    /// Read de-interleaved samples into per-channel buffers. Returns frames
    /// read.
    pub fn read_non_interleaved(&self, channel_data: &mut [&mut [f32]]) -> usize {
        if channel_data.is_empty() || self.channels == 0 {
            return 0;
        }
        let frames = channel_data.iter().map(|c| c.len()).min().unwrap_or(0);
        let mut read_frames = 0;
        for f in 0..frames {
            let mut frame_ok = true;
            for ch in channel_data.iter_mut() {
                if let Some(s) = self.inner.pop() {
                    ch[f] = s;
                } else {
                    frame_ok = false;
                    break;
                }
            }
            if frame_ok {
                read_frames += 1;
            } else {
                break;
            }
        }
        read_frames
    }

    fn update_meter(&self, sample: f32) {
        let abs = sample.abs();
        loop {
            let current = f64::from_bits(self.peak_level.load(Ordering::Relaxed));
            if abs as f64 <= current {
                break;
            }
            if self
                .peak_level
                .compare_exchange_weak(
                    current.to_bits(),
                    (abs as f64).to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }
        let sq = (sample * sample) as f64;
        loop {
            let old = f64::from_bits(self.rms_accumulator.load(Ordering::Relaxed));
            let new = old + sq;
            if self
                .rms_accumulator
                .compare_exchange_weak(
                    old.to_bits(),
                    new.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }
        self.rms_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Current peak level.
    pub fn peak_level(&self) -> f32 {
        f64::from_bits(self.peak_level.load(Ordering::Relaxed)) as f32
    }

    /// Current RMS level (resets accumulator).
    pub fn rms_level(&self) -> f32 {
        let acc = f64::from_bits(self.rms_accumulator.swap(0f64.to_bits(), Ordering::Relaxed));
        let count = self.rms_count.swap(0, Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            (acc / count as f64).sqrt() as f32
        }
    }

    /// Reset peak and RMS meters.
    pub fn reset_meters(&self) {
        self.peak_level.store(0f64.to_bits(), Ordering::Relaxed);
        self.rms_accumulator.store(0f64.to_bits(), Ordering::Relaxed);
        self.rms_count.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// EventQueue<T> — lock-free timestamped event queue
// ---------------------------------------------------------------------------

/// Policy applied when the event queue is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowPolicy {
    /// Drop the oldest event to make room.
    DropOldest,
    /// Drop the incoming event.
    DropNewest,
    /// Spin-wait until space is available (use with care).
    Block,
}

/// A timestamped event for parameter/control changes.
#[derive(Debug, Clone)]
pub struct TimestampedEvent<T> {
    pub timestamp_samples: u64,
    pub payload: T,
}

/// Lock-free bounded event queue with configurable overflow policy.
pub struct EventQueue<T> {
    inner: RingBuffer<TimestampedEvent<T>>,
    policy: OverflowPolicy,
    dropped: AtomicUsize,
}

impl<T> EventQueue<T> {
    pub fn new(capacity: usize, policy: OverflowPolicy) -> Self {
        Self {
            inner: RingBuffer::new(capacity),
            policy,
            dropped: AtomicUsize::new(0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Enqueue an event. Behaviour on full queue depends on `OverflowPolicy`.
    pub fn enqueue(&self, timestamp_samples: u64, payload: T) -> bool {
        let evt = TimestampedEvent {
            timestamp_samples,
            payload,
        };
        match self.try_enqueue_inner(evt) {
            Ok(()) => true,
            Err(evt) => self.handle_overflow(evt),
        }
    }

    fn try_enqueue_inner(&self, evt: TimestampedEvent<T>) -> Result<(), TimestampedEvent<T>> {
        self.inner.try_push(evt)
    }

    fn handle_overflow(&self, evt: TimestampedEvent<T>) -> bool {
        match self.policy {
            OverflowPolicy::DropNewest => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
            OverflowPolicy::DropOldest => {
                let _ = self.inner.pop();
                self.dropped.fetch_add(1, Ordering::Relaxed);
                self.inner.push(evt)
            }
            OverflowPolicy::Block => {
                let mut current = evt;
                loop {
                    match self.inner.try_push(current) {
                        Ok(()) => return true,
                        Err(e) => {
                            std::hint::spin_loop();
                            current = e;
                        }
                    }
                }
            }
        }
    }

    /// Dequeue the next event, if any.
    pub fn dequeue(&self) -> Option<TimestampedEvent<T>> {
        self.inner.pop()
    }

    /// Number of events currently in the queue.
    pub fn len(&self) -> usize {
        self.inner.available_read()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of events dropped due to overflow since last call.
    pub fn dropped_count(&self) -> usize {
        self.dropped.swap(0, Ordering::Relaxed)
    }

    /// Drain all events into a vector ordered by timestamp.
    pub fn drain_sorted(&self) -> Vec<TimestampedEvent<T>>
    where
        T: Clone,
    {
        let mut events = Vec::new();
        while let Some(e) = self.dequeue() {
            events.push(e);
        }
        events.sort_by_key(|e| e.timestamp_samples);
        events
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_power_of_two() {
        let rb: RingBuffer<i32> = RingBuffer::new(5);
        assert_eq!(rb.capacity(), 8);
        let rb2: RingBuffer<i32> = RingBuffer::new(8);
        assert_eq!(rb2.capacity(), 8);
    }

    #[test]
    fn ring_buffer_push_pop() {
        let rb: RingBuffer<i32> = RingBuffer::new(4);
        assert!(rb.is_empty());
        assert!(rb.push(1));
        assert!(rb.push(2));
        assert!(rb.push(3));
        assert!(rb.push(4));
        assert!(rb.is_full());
        assert!(!rb.push(5));
        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), Some(4));
        assert_eq!(rb.pop(), None);
    }

    #[test]
    fn ring_buffer_available_counts() {
        let rb: RingBuffer<u8> = RingBuffer::new(4);
        assert_eq!(rb.available_read(), 0);
        assert_eq!(rb.available_write(), 4);
        rb.push(10);
        rb.push(20);
        assert_eq!(rb.available_read(), 2);
        assert_eq!(rb.available_write(), 2);
    }

    #[test]
    fn ring_buffer_wrap_around() {
        let rb: RingBuffer<i32> = RingBuffer::new(4);
        for i in 0..4 {
            rb.push(i);
        }
        for _ in 0..2 {
            rb.pop();
        }
        rb.push(100);
        rb.push(200);
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), Some(100));
        assert_eq!(rb.pop(), Some(200));
    }

    #[test]
    fn ring_buffer_clear() {
        let rb: RingBuffer<i32> = RingBuffer::new(4);
        rb.push(1);
        rb.push(2);
        rb.clear();
        assert!(rb.is_empty());
    }

    #[test]
    fn triple_buffer_basic() {
        let tb = TripleBuffer::new(0i32);
        *tb.write_buf() = 42;
        tb.publish();
        let (val, fresh) = tb.read();
        assert_eq!(*val, 42);
        assert!(fresh);
        let (_, fresh2) = tb.read();
        assert!(!fresh2);
    }

    #[test]
    fn triple_buffer_overwrite() {
        let tb = TripleBuffer::new(0);
        *tb.write_buf() = 1;
        tb.publish();
        *tb.write_buf() = 2;
        tb.publish();
        let (val, _) = tb.read();
        assert_eq!(*val, 2);
    }

    #[test]
    fn audio_ring_buffer_interleaved() {
        let arb = AudioRingBuffer::new(64, 2);
        let samples = vec![0.5f32, -0.5, 0.3, -0.3];
        let written = arb.write_interleaved(&samples);
        assert_eq!(written, 4);
        let mut out = vec![0.0f32; 4];
        let read = arb.read_interleaved(&mut out);
        assert_eq!(read, 4);
        assert_eq!(out, samples);
    }

    #[test]
    fn audio_ring_buffer_metering() {
        let arb = AudioRingBuffer::new(64, 1);
        arb.write_interleaved(&[0.5, -0.8, 0.3]);
        assert!((arb.peak_level() - 0.8).abs() < 1e-5);
        let rms = arb.rms_level();
        assert!(rms > 0.0);
    }

    #[test]
    fn audio_ring_buffer_non_interleaved() {
        let arb = AudioRingBuffer::new(64, 2);
        let ch0: &[f32] = &[1.0, 2.0];
        let ch1: &[f32] = &[3.0, 4.0];
        let written = arb.write_non_interleaved(&[ch0, ch1]);
        assert_eq!(written, 2);
        let mut o0 = [0.0f32; 2];
        let mut o1 = [0.0f32; 2];
        let read = arb.read_non_interleaved(&mut [&mut o0[..], &mut o1[..]]);
        assert_eq!(read, 2);
        assert_eq!(o0, [1.0, 2.0]);
        assert_eq!(o1, [3.0, 4.0]);
    }

    #[test]
    fn event_queue_drop_newest() {
        let eq: EventQueue<i32> = EventQueue::new(2, OverflowPolicy::DropNewest);
        assert!(eq.enqueue(0, 10));
        assert!(eq.enqueue(1, 20));
        assert!(!eq.enqueue(2, 30));
        assert_eq!(eq.dropped_count(), 1);
        assert_eq!(eq.dequeue().unwrap().payload, 10);
    }

    #[test]
    fn event_queue_drop_oldest() {
        let eq: EventQueue<i32> = EventQueue::new(2, OverflowPolicy::DropOldest);
        eq.enqueue(0, 10);
        eq.enqueue(1, 20);
        eq.enqueue(2, 30);
        assert_eq!(eq.dropped_count(), 1);
        assert_eq!(eq.dequeue().unwrap().payload, 20);
        assert_eq!(eq.dequeue().unwrap().payload, 30);
    }

    #[test]
    fn event_queue_drain_sorted() {
        let eq: EventQueue<&str> = EventQueue::new(8, OverflowPolicy::DropNewest);
        eq.enqueue(10, "b");
        eq.enqueue(5, "a");
        eq.enqueue(20, "c");
        let sorted = eq.drain_sorted();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].timestamp_samples, 5);
        assert_eq!(sorted[2].timestamp_samples, 20);
    }

    #[test]
    fn event_queue_len_and_empty() {
        let eq: EventQueue<i32> = EventQueue::new(4, OverflowPolicy::DropNewest);
        assert!(eq.is_empty());
        eq.enqueue(0, 1);
        assert_eq!(eq.len(), 1);
        eq.dequeue();
        assert!(eq.is_empty());
    }
}
