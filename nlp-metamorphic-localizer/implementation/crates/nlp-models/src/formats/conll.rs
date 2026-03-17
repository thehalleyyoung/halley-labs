//! CoNLL-U format reader and writer.
//!
//! Implements the 10-column [CoNLL-U format](https://universaldependencies.org/format.html):
//! `ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC`
//!
//! Sentences are separated by blank lines.  Comment lines starting with `#`
//! are preserved on read and reproduced on write.

use shared_types::{
    DependencyEdge, DependencyRelation, LocalizerError, PosTag, Result, Sentence,
    Token,
};
use std::io::{BufRead, Write};
use std::path::Path;

// ── CoNLL-U sentence ────────────────────────────────────────────────────────

/// A single parsed CoNLL-U sentence with its comment lines.
#[derive(Debug, Clone)]
pub struct CoNLLSentence {
    /// Lines beginning with `#` that precede the token rows.
    pub comments: Vec<String>,
    /// Parsed token rows (1-indexed ID in the file).
    pub rows: Vec<CoNLLRow>,
}

/// One row of a CoNLL-U file (a single token).
#[derive(Debug, Clone)]
pub struct CoNLLRow {
    pub id: usize,
    pub form: String,
    pub lemma: String,
    pub upos: String,
    pub xpos: String,
    pub feats: String,
    pub head: usize,
    pub deprel: String,
    pub deps: String,
    pub misc: String,
}

// ── Reader ──────────────────────────────────────────────────────────────────

/// Reads CoNLL-U formatted data from files or strings.
pub struct CoNLLReader;

impl CoNLLReader {
    /// Read all sentences from a CoNLL-U file on disk.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Vec<CoNLLSentence>> {
        let file = std::fs::File::open(path.as_ref())?;
        let reader = std::io::BufReader::new(file);
        Self::read(reader)
    }

    /// Read all sentences from any [`BufRead`] source.
    pub fn read<R: BufRead>(reader: R) -> Result<Vec<CoNLLSentence>> {
        let mut sentences = Vec::new();
        let mut comments = Vec::new();
        let mut rows = Vec::new();

        for line_result in reader.lines() {
            let line = line_result?;
            let trimmed = line.trim();

            if trimmed.is_empty() {
                if !rows.is_empty() {
                    sentences.push(CoNLLSentence {
                        comments: std::mem::take(&mut comments),
                        rows: std::mem::take(&mut rows),
                    });
                }
                continue;
            }

            if trimmed.starts_with('#') {
                comments.push(trimmed.to_string());
                continue;
            }

            let row = Self::parse_row(trimmed)?;
            rows.push(row);
        }

        // Flush trailing sentence without a final blank line.
        if !rows.is_empty() {
            sentences.push(CoNLLSentence { comments, rows });
        }

        Ok(sentences)
    }

    /// Parse a CoNLL-U string (convenience wrapper around [`read`]).
    pub fn read_string(text: &str) -> Result<Vec<CoNLLSentence>> {
        Self::read(std::io::Cursor::new(text))
    }

    /// Convert a [`CoNLLSentence`] into the workspace [`Sentence`] type.
    pub fn to_sentence(conll: &CoNLLSentence) -> Sentence {
        let mut tokens = Vec::with_capacity(conll.rows.len());
        let mut edges = Vec::with_capacity(conll.rows.len());

        for row in &conll.rows {
            let mut tok = Token::new(&row.form, row.id - 1);
            if row.lemma != "_" {
                tok = tok.with_lemma(&row.lemma);
            }
            if row.upos != "_" {
                tok = tok.with_pos(Self::parse_upos(&row.upos));
            }
            // Store extra columns as features.
            if row.xpos != "_" {
                tok = tok.with_feature("xpos", &row.xpos);
            }
            if row.feats != "_" {
                tok = tok.with_feature("feats", &row.feats);
            }
            if row.deps != "_" {
                tok = tok.with_feature("deps", &row.deps);
            }
            if row.misc != "_" {
                tok = tok.with_feature("misc", &row.misc);
            }
            tokens.push(tok);

            edges.push(DependencyEdge::new(
                row.head,
                row.id,
                DependencyRelation::from_str_loose(&row.deprel),
            ));
        }

        let raw_text = tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join(" ");
        let mut sent = Sentence::from_tokens(tokens, raw_text);
        sent.dependency_edges = edges;
        sent
    }

    // ── private helpers ─────────────────────────────────────────────────────

    fn parse_row(line: &str) -> Result<CoNLLRow> {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 10 {
            return Err(LocalizerError::validation(
                "conll",
                format!(
                    "expected 10 tab-separated columns, got {} in line: {}",
                    parts.len(),
                    line
                ),
            ));
        }

        // Skip multi-word token ranges (e.g. "1-2") and empty nodes (e.g. "1.1").
        let id: usize = parts[0].parse().map_err(|_| {
            LocalizerError::validation("conll", format!("non-integer ID: {}", parts[0]))
        })?;
        let head: usize = parts[6].parse().map_err(|_| {
            LocalizerError::validation("conll", format!("non-integer HEAD: {}", parts[6]))
        })?;

        Ok(CoNLLRow {
            id,
            form: parts[1].to_string(),
            lemma: parts[2].to_string(),
            upos: parts[3].to_string(),
            xpos: parts[4].to_string(),
            feats: parts[5].to_string(),
            head,
            deprel: parts[7].to_string(),
            deps: parts[8].to_string(),
            misc: parts[9].to_string(),
        })
    }

    /// Best-effort mapping from UPOS tag strings to [`PosTag`].
    fn parse_upos(upos: &str) -> PosTag {
        match upos {
            "NOUN" | "PROPN" => PosTag::Noun,
            "VERB" => PosTag::Verb,
            "ADJ" => PosTag::Adj,
            "ADV" => PosTag::Adv,
            "DET" => PosTag::Det,
            "ADP" => PosTag::Prep,
            "CCONJ" | "SCONJ" => PosTag::Conj,
            "PRON" => PosTag::Pron,
            "AUX" => PosTag::Aux,
            "PUNCT" | "SYM" => PosTag::Punct,
            "NUM" => PosTag::Num,
            "PART" => PosTag::Part,
            "INTJ" => PosTag::Intj,
            _ => PosTag::Other,
        }
    }
}

// ── Writer ──────────────────────────────────────────────────────────────────

/// Writes data in CoNLL-U format to files or any [`Write`] sink.
pub struct CoNLLWriter;

impl CoNLLWriter {
    /// Write sentences to a file on disk.
    pub fn write_file(path: impl AsRef<Path>, sentences: &[CoNLLSentence]) -> Result<()> {
        let file = std::fs::File::create(path.as_ref())?;
        let mut writer = std::io::BufWriter::new(file);
        Self::write(&mut writer, sentences)
    }

    /// Write sentences to any [`Write`] sink.
    pub fn write<W: Write>(writer: &mut W, sentences: &[CoNLLSentence]) -> Result<()> {
        for (i, sent) in sentences.iter().enumerate() {
            for comment in &sent.comments {
                writeln!(writer, "{}", comment)?;
            }
            for row in &sent.rows {
                writeln!(
                    writer,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    row.id, row.form, row.lemma, row.upos, row.xpos, row.feats, row.head,
                    row.deprel, row.deps, row.misc,
                )?;
            }
            if i + 1 < sentences.len() {
                writeln!(writer)?;
            }
        }
        Ok(())
    }

    /// Render sentences to a [`String`].
    pub fn write_string(sentences: &[CoNLLSentence]) -> Result<String> {
        let mut buf = Vec::new();
        Self::write(&mut buf, sentences)?;
        String::from_utf8(buf).map_err(|e| {
            LocalizerError::validation("conll", format!("non-UTF-8 output: {e}"))
        })
    }

    /// Convert a workspace [`Sentence`] into a [`CoNLLSentence`].
    pub fn from_sentence(sent: &Sentence) -> CoNLLSentence {
        let mut head_map: std::collections::HashMap<usize, (usize, &DependencyRelation)> =
            std::collections::HashMap::new();
        for edge in &sent.dependency_edges {
            head_map.insert(edge.dependent_index, (edge.head_index, &edge.relation));
        }

        let rows = sent
            .tokens
            .iter()
            .enumerate()
            .map(|(i, tok)| {
                let id = i + 1;
                let (head, rel) = head_map
                    .get(&id)
                    .map(|(h, r)| (*h, format!("{}", r)))
                    .unwrap_or((0, "dep".into()));

                CoNLLRow {
                    id,
                    form: tok.text.clone(),
                    lemma: tok.lemma.clone().unwrap_or_else(|| "_".into()),
                    upos: tok
                        .pos_tag
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| "_".into()),
                    xpos: tok
                        .features
                        .get("xpos")
                        .cloned()
                        .unwrap_or_else(|| "_".into()),
                    feats: tok
                        .features
                        .get("feats")
                        .cloned()
                        .unwrap_or_else(|| "_".into()),
                    head,
                    deprel: rel,
                    deps: tok
                        .features
                        .get("deps")
                        .cloned()
                        .unwrap_or_else(|| "_".into()),
                    misc: tok
                        .features
                        .get("misc")
                        .cloned()
                        .unwrap_or_else(|| "_".into()),
                }
            })
            .collect();

        CoNLLSentence {
            comments: Vec::new(),
            rows,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CONLL: &str = "\
# sent_id = 1
# text = The cat sat on the mat.
1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\t_
2\tcat\tcat\tNOUN\tNN\t_\t3\tnsubj\t_\t_
3\tsat\tsit\tVERB\tVBD\t_\t0\troot\t_\t_
4\ton\ton\tADP\tIN\t_\t6\tcase\t_\t_
5\tthe\tthe\tDET\tDT\t_\t6\tdet\t_\t_
6\tmat\tmat\tNOUN\tNN\t_\t3\tnmod\t_\tSpaceAfter=No
7\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_

# sent_id = 2
# text = Dogs run.
1\tDogs\tdog\tNOUN\tNNS\t_\t2\tnsubj\t_\t_
2\trun\trun\tVERB\tVBP\t_\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_
";

    #[test]
    fn test_read_multiple_sentences() {
        let sentences = CoNLLReader::read_string(SAMPLE_CONLL).unwrap();
        assert_eq!(sentences.len(), 2);

        assert_eq!(sentences[0].comments.len(), 2);
        assert_eq!(sentences[0].rows.len(), 7);
        assert_eq!(sentences[0].rows[0].form, "The");
        assert_eq!(sentences[0].rows[2].lemma, "sit");

        assert_eq!(sentences[1].rows.len(), 3);
        assert_eq!(sentences[1].rows[0].form, "Dogs");
    }

    #[test]
    fn test_roundtrip() {
        let sentences = CoNLLReader::read_string(SAMPLE_CONLL).unwrap();
        let output = CoNLLWriter::write_string(&sentences).unwrap();
        let reparsed = CoNLLReader::read_string(&output).unwrap();

        assert_eq!(sentences.len(), reparsed.len());
        for (a, b) in sentences.iter().zip(reparsed.iter()) {
            assert_eq!(a.rows.len(), b.rows.len());
            for (ra, rb) in a.rows.iter().zip(b.rows.iter()) {
                assert_eq!(ra.form, rb.form);
                assert_eq!(ra.lemma, rb.lemma);
                assert_eq!(ra.head, rb.head);
                assert_eq!(ra.deprel, rb.deprel);
            }
        }
    }

    #[test]
    fn test_to_sentence_conversion() {
        let sentences = CoNLLReader::read_string(SAMPLE_CONLL).unwrap();
        let sent = CoNLLReader::to_sentence(&sentences[0]);

        assert_eq!(sent.tokens.len(), 7);
        assert_eq!(sent.tokens[0].text, "The");
        assert_eq!(sent.tokens[0].pos_tag, Some(PosTag::Det));
        assert_eq!(sent.tokens[2].lemma.as_deref(), Some("sit"));
        assert_eq!(sent.dependency_edges.len(), 7);
    }

    #[test]
    fn test_from_sentence_conversion() {
        let sentences = CoNLLReader::read_string(SAMPLE_CONLL).unwrap();
        let sent = CoNLLReader::to_sentence(&sentences[0]);
        let conll = CoNLLWriter::from_sentence(&sent);

        assert_eq!(conll.rows.len(), 7);
        assert_eq!(conll.rows[0].form, "The");
        assert_eq!(conll.rows[0].upos, "DET");
    }

    #[test]
    fn test_parse_row_too_few_columns() {
        let result = CoNLLReader::read_string("1\tThe\tthe\tDET\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input() {
        let sentences = CoNLLReader::read_string("").unwrap();
        assert!(sentences.is_empty());
    }

    #[test]
    fn test_write_string_single_sentence() {
        let sentences = CoNLLReader::read_string(SAMPLE_CONLL).unwrap();
        let out = CoNLLWriter::write_string(&sentences[..1]).unwrap();
        assert!(out.contains("The\tthe\tDET"));
        assert!(out.contains("# sent_id = 1"));
    }
}
