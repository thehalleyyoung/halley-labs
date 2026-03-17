//! TPC-C workload modelling.
//!
//! Models the five standard TPC-C transaction profiles (New-Order, Payment,
//! Order-Status, Delivery, Stock-Level) as IR programs suitable for
//! IsoSpec analysis.

use std::collections::HashMap;

use isospec_types::column::DataType;
use isospec_types::identifier::{IdAllocator, TableId, TransactionId, WorkloadId};
use isospec_types::ir::{
    IrDelete, IrExpr, IrInsert, IrProgram, IrSelect, IrStatement, IrTransaction, IrUpdate,
};
use isospec_types::isolation::IsolationLevel;
use isospec_types::predicate::{ColumnRef, Predicate};
use isospec_types::schema::{ColumnDef, ForeignKeyDef, IndexDef, IndexColumn, Schema, TableSchema};
use isospec_types::value::Value;
use isospec_types::workload::{Workload, WorkloadParameters};

// ---------------------------------------------------------------------------
// TPC-C Schema
// ---------------------------------------------------------------------------

/// Build a simplified TPC-C schema with all nine tables.
pub fn tpcc_schema() -> Schema {
    let mut schema = Schema::new();
    let mut tid_alloc: IdAllocator<TableId> = IdAllocator::new();

    // helper closures
    let int_col = |name: &str, pk: bool| -> ColumnDef {
        let mut cd = ColumnDef {
            name: name.to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: pk,
            unique: pk,
            default: None,
            auto_increment: false,
            references: None,
        };
        if pk { cd = cd.not_null().primary_key(); }
        cd
    };
    let varchar_col = |name: &str, len: u32| -> ColumnDef {
        ColumnDef {
            name: name.to_string(),
            data_type: DataType::Varchar(len),
            nullable: true,
            primary_key: false,
            unique: false,
            default: None,
            auto_increment: false,
            references: None,
        }
    };
    let decimal_col = |name: &str| -> ColumnDef {
        ColumnDef {
            name: name.to_string(),
            data_type: DataType::Decimal { precision: 12, scale: 2 },
            nullable: false,
            primary_key: false,
            unique: false,
            default: None,
            auto_increment: false,
            references: None,
        }
    };
    let ts_col = |name: &str| -> ColumnDef {
        ColumnDef {
            name: name.to_string(),
            data_type: DataType::Timestamp,
            nullable: true,
            primary_key: false,
            unique: false,
            default: None,
            auto_increment: false,
            references: None,
        }
    };

    // WAREHOUSE
    let wh_id = tid_alloc.allocate();
    let wh = TableSchema::new(wh_id, "warehouse".to_string())
        .with_column(int_col("w_id", true))
        .with_column(varchar_col("w_name", 10))
        .with_column(decimal_col("w_tax"))
        .with_column(decimal_col("w_ytd"));
    schema.add_table(wh);

    // DISTRICT
    let dist_id = tid_alloc.allocate();
    let dist = TableSchema::new(dist_id, "district".to_string())
        .with_column(int_col("d_id", true))
        .with_column(int_col("d_w_id", true))
        .with_column(varchar_col("d_name", 10))
        .with_column(decimal_col("d_tax"))
        .with_column(decimal_col("d_ytd"))
        .with_column(int_col("d_next_o_id", false));
    schema.add_table(dist);

    // CUSTOMER
    let cust_id = tid_alloc.allocate();
    let cust = TableSchema::new(cust_id, "customer".to_string())
        .with_column(int_col("c_id", true))
        .with_column(int_col("c_d_id", true))
        .with_column(int_col("c_w_id", true))
        .with_column(varchar_col("c_last", 16))
        .with_column(decimal_col("c_balance"))
        .with_column(decimal_col("c_ytd_payment"))
        .with_column(int_col("c_payment_cnt", false));
    schema.add_table(cust);

    // ITEM
    let item_id = tid_alloc.allocate();
    let item = TableSchema::new(item_id, "item".to_string())
        .with_column(int_col("i_id", true))
        .with_column(varchar_col("i_name", 24))
        .with_column(decimal_col("i_price"))
        .with_column(int_col("i_im_id", false));
    schema.add_table(item);

    // STOCK
    let stock_id = tid_alloc.allocate();
    let stock = TableSchema::new(stock_id, "stock".to_string())
        .with_column(int_col("s_i_id", true))
        .with_column(int_col("s_w_id", true))
        .with_column(int_col("s_quantity", false))
        .with_column(int_col("s_order_cnt", false))
        .with_column(int_col("s_remote_cnt", false));
    schema.add_table(stock);

    // ORDER (named "orders" to avoid SQL keyword)
    let ord_id = tid_alloc.allocate();
    let ord = TableSchema::new(ord_id, "orders".to_string())
        .with_column(int_col("o_id", true))
        .with_column(int_col("o_d_id", true))
        .with_column(int_col("o_w_id", true))
        .with_column(int_col("o_c_id", false))
        .with_column(ts_col("o_entry_d"))
        .with_column(int_col("o_ol_cnt", false));
    schema.add_table(ord);

    // ORDER-LINE
    let ol_id = tid_alloc.allocate();
    let ol = TableSchema::new(ol_id, "order_line".to_string())
        .with_column(int_col("ol_o_id", true))
        .with_column(int_col("ol_d_id", true))
        .with_column(int_col("ol_w_id", true))
        .with_column(int_col("ol_number", true))
        .with_column(int_col("ol_i_id", false))
        .with_column(int_col("ol_quantity", false))
        .with_column(decimal_col("ol_amount"))
        .with_column(ts_col("ol_delivery_d"));
    schema.add_table(ol);

    // NEW-ORDER
    let no_id = tid_alloc.allocate();
    let no = TableSchema::new(no_id, "new_order".to_string())
        .with_column(int_col("no_o_id", true))
        .with_column(int_col("no_d_id", true))
        .with_column(int_col("no_w_id", true));
    schema.add_table(no);

    // HISTORY
    let hist_id = tid_alloc.allocate();
    let hist = TableSchema::new(hist_id, "history".to_string())
        .with_column(int_col("h_c_id", false))
        .with_column(int_col("h_c_d_id", false))
        .with_column(int_col("h_c_w_id", false))
        .with_column(int_col("h_d_id", false))
        .with_column(int_col("h_w_id", false))
        .with_column(decimal_col("h_amount"))
        .with_column(ts_col("h_date"));
    schema.add_table(hist);

    schema
}

// ---------------------------------------------------------------------------
// Transaction templates
// ---------------------------------------------------------------------------

/// Generate a New-Order transaction for the given warehouse/district/order.
pub fn new_order_txn(
    txn_id: TransactionId,
    w_id: i64,
    d_id: i64,
    o_id: i64,
    item_ids: &[i64],
    level: IsolationLevel,
) -> IrTransaction {
    let mut stmts = Vec::new();

    // 1. Read district to get next order id
    stmts.push(select("district", &["d_next_o_id", "d_tax"],
        Predicate::And(vec![
            Predicate::eq("d_id", Value::Integer(d_id)),
            Predicate::eq("d_w_id", Value::Integer(w_id)),
        ])));

    // 2. Increment d_next_o_id
    stmts.push(update("district", &[("d_next_o_id", IrExpr::Literal(Value::Integer(o_id + 1)))],
        Predicate::And(vec![
            Predicate::eq("d_id", Value::Integer(d_id)),
            Predicate::eq("d_w_id", Value::Integer(w_id)),
        ])));

    // 3. Insert new order
    stmts.push(insert("orders",
        &["o_id", "o_d_id", "o_w_id", "o_c_id", "o_ol_cnt"],
        vec![lit(o_id), lit(d_id), lit(w_id), lit(1), lit(item_ids.len() as i64)],
    ));

    // 4. Insert new-order marker
    stmts.push(insert("new_order",
        &["no_o_id", "no_d_id", "no_w_id"],
        vec![lit(o_id), lit(d_id), lit(w_id)],
    ));

    // 5. For each item: read item price, update stock, insert order-line
    for (ol_num, &i_id) in item_ids.iter().enumerate() {
        stmts.push(select("item", &["i_price", "i_name"],
            Predicate::eq("i_id", Value::Integer(i_id))));
        stmts.push(select("stock", &["s_quantity"],
            Predicate::And(vec![
                Predicate::eq("s_i_id", Value::Integer(i_id)),
                Predicate::eq("s_w_id", Value::Integer(w_id)),
            ])));
        stmts.push(update("stock",
            &[("s_quantity", IrExpr::Literal(Value::Integer(90))),
              ("s_order_cnt", IrExpr::Literal(Value::Integer(1)))],
            Predicate::And(vec![
                Predicate::eq("s_i_id", Value::Integer(i_id)),
                Predicate::eq("s_w_id", Value::Integer(w_id)),
            ])));
        stmts.push(insert("order_line",
            &["ol_o_id", "ol_d_id", "ol_w_id", "ol_number", "ol_i_id", "ol_quantity"],
            vec![lit(o_id), lit(d_id), lit(w_id), lit(ol_num as i64 + 1), lit(i_id), lit(5)],
        ));
    }

    IrTransaction {
        id: txn_id,
        label: format!("NewOrder_w{}_d{}_o{}", w_id, d_id, o_id),
        isolation_level: level,
        statements: stmts,
        read_only: false,
    }
}

/// Payment transaction template.
pub fn payment_txn(
    txn_id: TransactionId,
    w_id: i64,
    d_id: i64,
    c_id: i64,
    amount: i64,
    level: IsolationLevel,
) -> IrTransaction {
    let mut stmts = Vec::new();

    // Update warehouse YTD
    stmts.push(update("warehouse",
        &[("w_ytd", IrExpr::Literal(Value::Integer(amount)))],
        Predicate::eq("w_id", Value::Integer(w_id)),
    ));

    // Update district YTD
    stmts.push(update("district",
        &[("d_ytd", IrExpr::Literal(Value::Integer(amount)))],
        Predicate::And(vec![
            Predicate::eq("d_id", Value::Integer(d_id)),
            Predicate::eq("d_w_id", Value::Integer(w_id)),
        ]),
    ));

    // Read customer
    stmts.push(select("customer",
        &["c_balance", "c_ytd_payment", "c_payment_cnt"],
        Predicate::And(vec![
            Predicate::eq("c_id", Value::Integer(c_id)),
            Predicate::eq("c_d_id", Value::Integer(d_id)),
            Predicate::eq("c_w_id", Value::Integer(w_id)),
        ]),
    ));

    // Update customer balance
    stmts.push(update("customer",
        &[("c_balance", IrExpr::Literal(Value::Integer(-amount))),
          ("c_ytd_payment", IrExpr::Literal(Value::Integer(amount))),
          ("c_payment_cnt", IrExpr::Literal(Value::Integer(1)))],
        Predicate::And(vec![
            Predicate::eq("c_id", Value::Integer(c_id)),
            Predicate::eq("c_d_id", Value::Integer(d_id)),
            Predicate::eq("c_w_id", Value::Integer(w_id)),
        ]),
    ));

    // Insert history row
    stmts.push(insert("history",
        &["h_c_id", "h_c_d_id", "h_c_w_id", "h_d_id", "h_w_id", "h_amount"],
        vec![lit(c_id), lit(d_id), lit(w_id), lit(d_id), lit(w_id), lit(amount)],
    ));

    IrTransaction {
        id: txn_id,
        label: format!("Payment_w{}_d{}_c{}", w_id, d_id, c_id),
        isolation_level: level,
        statements: stmts,
        read_only: false,
    }
}

/// Order-Status transaction (read-only).
pub fn order_status_txn(
    txn_id: TransactionId,
    w_id: i64,
    d_id: i64,
    c_id: i64,
    level: IsolationLevel,
) -> IrTransaction {
    let mut stmts = Vec::new();

    // Read customer
    stmts.push(select("customer", &["c_balance", "c_last"],
        Predicate::And(vec![
            Predicate::eq("c_id", Value::Integer(c_id)),
            Predicate::eq("c_d_id", Value::Integer(d_id)),
            Predicate::eq("c_w_id", Value::Integer(w_id)),
        ])));

    // Read last order
    stmts.push(select("orders", &["o_id", "o_entry_d", "o_ol_cnt"],
        Predicate::And(vec![
            Predicate::eq("o_c_id", Value::Integer(c_id)),
            Predicate::eq("o_d_id", Value::Integer(d_id)),
            Predicate::eq("o_w_id", Value::Integer(w_id)),
        ])));

    // Read order lines
    stmts.push(select("order_line",
        &["ol_i_id", "ol_quantity", "ol_amount", "ol_delivery_d"],
        Predicate::And(vec![
            Predicate::eq("ol_d_id", Value::Integer(d_id)),
            Predicate::eq("ol_w_id", Value::Integer(w_id)),
        ])));

    IrTransaction {
        id: txn_id,
        label: format!("OrderStatus_w{}_d{}_c{}", w_id, d_id, c_id),
        isolation_level: level,
        statements: stmts,
        read_only: true,
    }
}

/// Delivery transaction template.
pub fn delivery_txn(
    txn_id: TransactionId,
    w_id: i64,
    d_id: i64,
    o_id: i64,
    level: IsolationLevel,
) -> IrTransaction {
    let mut stmts = Vec::new();

    // Select oldest new-order
    stmts.push(select("new_order", &["no_o_id"],
        Predicate::And(vec![
            Predicate::eq("no_d_id", Value::Integer(d_id)),
            Predicate::eq("no_w_id", Value::Integer(w_id)),
        ])));

    // Delete from new_order
    stmts.push(IrStatement::Delete(IrDelete {
        table: "new_order".to_string(),
        predicate: Predicate::And(vec![
            Predicate::eq("no_o_id", Value::Integer(o_id)),
            Predicate::eq("no_d_id", Value::Integer(d_id)),
            Predicate::eq("no_w_id", Value::Integer(w_id)),
        ]),
    }));

    // Read order to get customer id
    stmts.push(select("orders", &["o_c_id"],
        Predicate::And(vec![
            Predicate::eq("o_id", Value::Integer(o_id)),
            Predicate::eq("o_d_id", Value::Integer(d_id)),
            Predicate::eq("o_w_id", Value::Integer(w_id)),
        ])));

    // Update order-lines with delivery date
    stmts.push(update("order_line",
        &[("ol_delivery_d", IrExpr::Literal(Value::Integer(1000)))],
        Predicate::And(vec![
            Predicate::eq("ol_o_id", Value::Integer(o_id)),
            Predicate::eq("ol_d_id", Value::Integer(d_id)),
            Predicate::eq("ol_w_id", Value::Integer(w_id)),
        ])));

    // Update customer balance (aggregate amount from order-lines)
    stmts.push(update("customer",
        &[("c_balance", IrExpr::Literal(Value::Integer(100)))],
        Predicate::And(vec![
            Predicate::eq("c_d_id", Value::Integer(d_id)),
            Predicate::eq("c_w_id", Value::Integer(w_id)),
        ])));

    IrTransaction {
        id: txn_id,
        label: format!("Delivery_w{}_d{}_o{}", w_id, d_id, o_id),
        isolation_level: level,
        statements: stmts,
        read_only: false,
    }
}

/// Stock-Level transaction (read-only).
pub fn stock_level_txn(
    txn_id: TransactionId,
    w_id: i64,
    d_id: i64,
    threshold: i64,
    level: IsolationLevel,
) -> IrTransaction {
    let mut stmts = Vec::new();

    // Read d_next_o_id
    stmts.push(select("district", &["d_next_o_id"],
        Predicate::And(vec![
            Predicate::eq("d_id", Value::Integer(d_id)),
            Predicate::eq("d_w_id", Value::Integer(w_id)),
        ])));

    // Read recent order-lines (predicate-range on ol_o_id)
    stmts.push(IrStatement::Select(IrSelect {
        table: "order_line".to_string(),
        columns: vec!["ol_i_id".to_string()],
        predicate: Predicate::And(vec![
            Predicate::eq("ol_d_id", Value::Integer(d_id)),
            Predicate::eq("ol_w_id", Value::Integer(w_id)),
        ]),
        for_update: false,
        for_share: false,
    }));

    // Read stock for items below threshold
    stmts.push(IrStatement::Select(IrSelect {
        table: "stock".to_string(),
        columns: vec!["s_i_id".to_string(), "s_quantity".to_string()],
        predicate: Predicate::And(vec![
            Predicate::eq("s_w_id", Value::Integer(w_id)),
            Predicate::lt("s_quantity", Value::Integer(threshold)),
        ]),
        for_update: false,
        for_share: false,
    }));

    IrTransaction {
        id: txn_id,
        label: format!("StockLevel_w{}_d{}_thr{}", w_id, d_id, threshold),
        isolation_level: level,
        statements: stmts,
        read_only: true,
    }
}

// ---------------------------------------------------------------------------
// Program builder
// ---------------------------------------------------------------------------

/// Build a full TPC-C IR program with all five transaction types.
pub fn tpcc_program(level: IsolationLevel) -> IrProgram {
    let mut txn_alloc: IdAllocator<TransactionId> = IdAllocator::new();

    let t1 = txn_alloc.allocate();
    let t2 = txn_alloc.allocate();
    let t3 = txn_alloc.allocate();
    let t4 = txn_alloc.allocate();
    let t5 = txn_alloc.allocate();

    let txns = vec![
        new_order_txn(t1, 1, 1, 3001, &[10, 20, 30], level),
        payment_txn(t2, 1, 1, 1, 5000, level),
        order_status_txn(t3, 1, 1, 1, level),
        delivery_txn(t4, 1, 1, 3000, level),
        stock_level_txn(t5, 1, 1, 10, level),
    ];

    IrProgram {
        id: WorkloadId::new(1),
        name: "tpcc_standard".to_string(),
        transactions: txns,
        schema_name: "tpcc".to_string(),
        metadata: HashMap::new(),
    }
}

/// Build a full TPC-C Workload.
pub fn tpcc_workload(level: IsolationLevel) -> Workload {
    let program = tpcc_program(level);
    let schema = tpcc_schema();
    Workload {
        id: program.id,
        name: program.name.clone(),
        program,
        schema,
        parameters: WorkloadParameters {
            transaction_bound: 5,
            operation_bound: 20,
            data_item_bound: 50,
            repetitions: 1,
        },
        annotations: HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// Statement helpers
// ---------------------------------------------------------------------------

fn select(table: &str, cols: &[&str], pred: Predicate) -> IrStatement {
    IrStatement::Select(IrSelect {
        table: table.to_string(),
        columns: cols.iter().map(|c| c.to_string()).collect(),
        predicate: pred,
        for_update: false,
        for_share: false,
    })
}

fn update(table: &str, assignments: &[(&str, IrExpr)], pred: Predicate) -> IrStatement {
    IrStatement::Update(IrUpdate {
        table: table.to_string(),
        assignments: assignments.iter().map(|(c, e)| (c.to_string(), e.clone())).collect(),
        predicate: pred,
    })
}

fn insert(table: &str, cols: &[&str], vals: Vec<IrExpr>) -> IrStatement {
    IrStatement::Insert(IrInsert {
        table: table.to_string(),
        columns: cols.iter().map(|c| c.to_string()).collect(),
        values: vec![vals],
    })
}

fn lit(v: i64) -> IrExpr {
    IrExpr::Literal(Value::Integer(v))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpcc_schema_tables() {
        let schema = tpcc_schema();
        let names = schema.table_names();
        assert!(names.contains(&"warehouse".to_string()));
        assert!(names.contains(&"district".to_string()));
        assert!(names.contains(&"customer".to_string()));
        assert!(names.contains(&"item".to_string()));
        assert!(names.contains(&"stock".to_string()));
        assert!(names.contains(&"orders".to_string()));
        assert!(names.contains(&"order_line".to_string()));
        assert!(names.contains(&"new_order".to_string()));
        assert!(names.contains(&"history".to_string()));
        assert_eq!(names.len(), 9);
    }

    #[test]
    fn test_new_order_txn() {
        let txn = new_order_txn(TransactionId::new(1), 1, 1, 3001, &[10, 20], IsolationLevel::Serializable);
        assert!(!txn.read_only);
        assert!(txn.has_writes());
        // 2 (district read+update) + 1 order insert + 1 new_order insert + 2*4 item ops = 12
        assert!(txn.statement_count() >= 10);
    }

    #[test]
    fn test_payment_txn() {
        let txn = payment_txn(TransactionId::new(2), 1, 1, 1, 5000, IsolationLevel::ReadCommitted);
        assert!(!txn.read_only);
        assert!(txn.has_writes());
        assert_eq!(txn.statement_count(), 5);
    }

    #[test]
    fn test_order_status_txn() {
        let txn = order_status_txn(TransactionId::new(3), 1, 1, 1, IsolationLevel::RepeatableRead);
        assert!(txn.read_only);
        assert!(!txn.has_writes());
    }

    #[test]
    fn test_delivery_txn() {
        let txn = delivery_txn(TransactionId::new(4), 1, 1, 3000, IsolationLevel::Serializable);
        assert!(!txn.read_only);
        assert!(txn.has_writes());
    }

    #[test]
    fn test_stock_level_txn() {
        let txn = stock_level_txn(TransactionId::new(5), 1, 1, 10, IsolationLevel::ReadCommitted);
        assert!(txn.read_only);
        assert!(!txn.has_writes());
        assert_eq!(txn.statement_count(), 3);
    }

    #[test]
    fn test_tpcc_program() {
        let prog = tpcc_program(IsolationLevel::Serializable);
        assert_eq!(prog.transaction_count(), 5);
        assert!(prog.total_statements() > 20);
    }

    #[test]
    fn test_tpcc_workload() {
        let wl = tpcc_workload(IsolationLevel::Snapshot);
        assert_eq!(wl.program.transaction_count(), 5);
        assert_eq!(wl.schema.table_names().len(), 9);
    }

    #[test]
    fn test_schema_warehouse_columns() {
        let schema = tpcc_schema();
        let wh = schema.get_table("warehouse").unwrap();
        let cols = wh.column_names();
        assert!(cols.contains(&"w_id".to_string()));
        assert!(cols.contains(&"w_tax".to_string()));
        assert!(cols.contains(&"w_ytd".to_string()));
    }

    #[test]
    fn test_schema_order_line_columns() {
        let schema = tpcc_schema();
        let ol = schema.get_table("order_line").unwrap();
        let cols = ol.column_names();
        assert!(cols.contains(&"ol_o_id".to_string()));
        assert!(cols.contains(&"ol_amount".to_string()));
    }
}
