//! Database schema definitions.
use serde::{Deserialize, Serialize};
pub use crate::column::{ColumnDef, DataType};
use crate::identifier::TableId;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub tables: HashMap<String, TableSchema>,
    pub table_id_map: HashMap<String, TableId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub id: TableId,
    pub name: String,
    pub columns: Vec<ColumnDef>,
    pub primary_key: Vec<String>,
    pub indexes: Vec<IndexDef>,
    pub unique_constraints: Vec<Vec<String>>,
    pub foreign_keys: Vec<ForeignKeyDef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDef {
    pub name: String,
    pub columns: Vec<IndexColumn>,
    pub unique: bool,
    pub index_type: IndexType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexColumn {
    pub name: String,
    pub ascending: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    BTree,
    Hash,
    GiST,
    GIN,
    Clustered,
    NonClustered,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKeyDef {
    pub name: String,
    pub columns: Vec<String>,
    pub ref_table: String,
    pub ref_columns: Vec<String>,
    pub on_delete: ReferentialAction,
    pub on_update: ReferentialAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferentialAction {
    NoAction,
    Cascade,
    SetNull,
    SetDefault,
    Restrict,
}

impl Schema {
    pub fn new() -> Self {
        Self { tables: HashMap::new(), table_id_map: HashMap::new() }
    }
    pub fn add_table(&mut self, table: TableSchema) {
        self.table_id_map.insert(table.name.clone(), table.id);
        self.tables.insert(table.name.clone(), table);
    }
    pub fn get_table(&self, name: &str) -> Option<&TableSchema> {
        self.tables.get(name)
    }
    pub fn table_id(&self, name: &str) -> Option<TableId> {
        self.table_id_map.get(name).copied()
    }
    pub fn table_names(&self) -> Vec<&str> {
        self.tables.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for Schema {
    fn default() -> Self { Self::new() }
}

impl TableSchema {
    pub fn new(id: TableId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            columns: Vec::new(),
            primary_key: Vec::new(),
            indexes: Vec::new(),
            unique_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        }
    }
    pub fn with_column(mut self, col: ColumnDef) -> Self {
        if col.primary_key {
            self.primary_key.push(col.name.clone());
        }
        self.columns.push(col);
        self
    }
    pub fn with_index(mut self, idx: IndexDef) -> Self {
        self.indexes.push(idx);
        self
    }
    pub fn get_column(&self, name: &str) -> Option<&ColumnDef> {
        self.columns.iter().find(|c| c.name == name)
    }
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }
    pub fn nullable_columns(&self) -> Vec<&ColumnDef> {
        self.columns.iter().filter(|c| c.nullable).collect()
    }
    pub fn has_index_on(&self, columns: &[&str]) -> bool {
        self.indexes.iter().any(|idx| {
            let idx_cols: Vec<&str> = idx.columns.iter().map(|c| c.name.as_str()).collect();
            columns.iter().all(|c| idx_cols.contains(c))
        })
    }
    pub fn applicable_indexes(&self, columns: &[&str]) -> Vec<&IndexDef> {
        self.indexes.iter().filter(|idx| {
            if idx.columns.is_empty() { return false; }
            let first = idx.columns[0].name.as_str();
            columns.contains(&first)
        }).collect()
    }
    pub fn row_size_estimate(&self) -> usize {
        self.columns.iter().map(|c| c.data_type.byte_size_estimate()).sum()
    }
}

pub fn tpcc_schema() -> Schema {
    let mut schema = Schema::new();
    schema.add_table(
        TableSchema::new(TableId::new(0), "warehouse")
            .with_column(ColumnDef::new("w_id", DataType::Integer).primary_key())
            .with_column(ColumnDef::new("w_name", DataType::Varchar(10)).not_null())
            .with_column(ColumnDef::new("w_ytd", DataType::Decimal { precision: 12, scale: 2 }).not_null())
    );
    schema.add_table(
        TableSchema::new(TableId::new(1), "district")
            .with_column(ColumnDef::new("d_id", DataType::Integer).primary_key())
            .with_column(ColumnDef::new("d_w_id", DataType::Integer).not_null().references("warehouse", "w_id"))
            .with_column(ColumnDef::new("d_name", DataType::Varchar(10)).not_null())
            .with_column(ColumnDef::new("d_ytd", DataType::Decimal { precision: 12, scale: 2 }).not_null())
            .with_column(ColumnDef::new("d_next_o_id", DataType::Integer).not_null())
    );
    schema.add_table(
        TableSchema::new(TableId::new(2), "customer")
            .with_column(ColumnDef::new("c_id", DataType::Integer).primary_key())
            .with_column(ColumnDef::new("c_d_id", DataType::Integer).not_null())
            .with_column(ColumnDef::new("c_w_id", DataType::Integer).not_null())
            .with_column(ColumnDef::new("c_balance", DataType::Decimal { precision: 12, scale: 2 }).not_null())
            .with_column(ColumnDef::new("c_ytd_payment", DataType::Decimal { precision: 12, scale: 2 }).not_null())
    );
    schema.add_table(
        TableSchema::new(TableId::new(3), "orders")
            .with_column(ColumnDef::new("o_id", DataType::Integer).primary_key())
            .with_column(ColumnDef::new("o_d_id", DataType::Integer).not_null())
            .with_column(ColumnDef::new("o_w_id", DataType::Integer).not_null())
            .with_column(ColumnDef::new("o_c_id", DataType::Integer).not_null())
            .with_column(ColumnDef::new("o_entry_d", DataType::Timestamp).not_null())
    );
    schema.add_table(
        TableSchema::new(TableId::new(4), "order_line")
            .with_column(ColumnDef::new("ol_o_id", DataType::Integer).primary_key())
            .with_column(ColumnDef::new("ol_d_id", DataType::Integer).not_null())
            .with_column(ColumnDef::new("ol_w_id", DataType::Integer).not_null())
            .with_column(ColumnDef::new("ol_number", DataType::Integer).not_null())
            .with_column(ColumnDef::new("ol_amount", DataType::Decimal { precision: 6, scale: 2 }).not_null())
    );
    schema.add_table(
        TableSchema::new(TableId::new(5), "stock")
            .with_column(ColumnDef::new("s_i_id", DataType::Integer).primary_key())
            .with_column(ColumnDef::new("s_w_id", DataType::Integer).not_null())
            .with_column(ColumnDef::new("s_quantity", DataType::Integer).not_null())
    );
    schema.add_table(
        TableSchema::new(TableId::new(6), "item")
            .with_column(ColumnDef::new("i_id", DataType::Integer).primary_key())
            .with_column(ColumnDef::new("i_name", DataType::Varchar(24)).not_null())
            .with_column(ColumnDef::new("i_price", DataType::Decimal { precision: 5, scale: 2 }).not_null())
    );
    schema
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_schema_creation() {
        let schema = tpcc_schema();
        assert!(schema.get_table("warehouse").is_some());
        assert!(schema.get_table("district").is_some());
        assert_eq!(schema.table_names().len(), 7);
    }
    #[test]
    fn test_table_column_lookup() {
        let schema = tpcc_schema();
        let warehouse = schema.get_table("warehouse").unwrap();
        assert!(warehouse.get_column("w_id").is_some());
        assert!(warehouse.get_column("nonexistent").is_none());
    }
    #[test]
    fn test_row_size() {
        let schema = tpcc_schema();
        let warehouse = schema.get_table("warehouse").unwrap();
        assert!(warehouse.row_size_estimate() > 0);
    }
}
