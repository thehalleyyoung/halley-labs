//! Simplified TPC-E workload modelling.
//!
//! Models the three core TPC-E transaction profiles (Trade-Order, Trade-Result,
//! Market-Feed) as IR programs for IsoSpec analysis. This is a simplified
//! subset that captures the essential concurrency characteristics.

use std::collections::HashMap;

use isospec_types::column::DataType;
use isospec_types::identifier::{IdAllocator, TableId, TransactionId, WorkloadId};
use isospec_types::ir::{
    IrExpr, IrInsert, IrProgram, IrSelect, IrStatement, IrTransaction, IrUpdate,
};
use isospec_types::isolation::IsolationLevel;
use isospec_types::predicate::Predicate;
use isospec_types::schema::{ColumnDef, Schema, TableSchema};
use isospec_types::value::Value;
use isospec_types::workload::{Workload, WorkloadParameters};

// ---------------------------------------------------------------------------
// TPC-E Schema (simplified subset)
// ---------------------------------------------------------------------------

/// Build a simplified TPC-E schema with the tables relevant to our three
/// transaction profiles.
pub fn tpce_schema() -> Schema {
    let mut schema = Schema::new();
    let mut tid_alloc: IdAllocator<TableId> = IdAllocator::new();

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
    let bigint_col = |name: &str| -> ColumnDef {
        ColumnDef {
            name: name.to_string(),
            data_type: DataType::BigInt,
            nullable: false,
            primary_key: false,
            unique: false,
            default: None,
            auto_increment: false,
            references: None,
        }
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

    // ACCOUNT
    let acc_id = tid_alloc.allocate();
    let acc = TableSchema::new(acc_id, "account".to_string())
        .with_column(int_col("ca_id", true))
        .with_column(int_col("ca_b_id", false))
        .with_column(int_col("ca_c_id", false))
        .with_column(varchar_col("ca_name", 50))
        .with_column(decimal_col("ca_bal"));
    schema.add_table(acc);

    // CUSTOMER
    let cust_id = tid_alloc.allocate();
    let cust = TableSchema::new(cust_id, "customer".to_string())
        .with_column(int_col("c_id", true))
        .with_column(varchar_col("c_l_name", 30))
        .with_column(varchar_col("c_f_name", 30))
        .with_column(int_col("c_tier", false));
    schema.add_table(cust);

    // BROKER
    let broker_id = tid_alloc.allocate();
    let broker = TableSchema::new(broker_id, "broker".to_string())
        .with_column(int_col("b_id", true))
        .with_column(varchar_col("b_name", 50))
        .with_column(int_col("b_num_trades", false))
        .with_column(decimal_col("b_comm_total"));
    schema.add_table(broker);

    // SECURITY
    let sec_id = tid_alloc.allocate();
    let sec = TableSchema::new(sec_id, "security".to_string())
        .with_column(int_col("s_symb", true))
        .with_column(varchar_col("s_name", 70))
        .with_column(int_col("s_ex_id", false))
        .with_column(bigint_col("s_num_out"));
    schema.add_table(sec);

    // TRADE
    let trade_id = tid_alloc.allocate();
    let trade = TableSchema::new(trade_id, "trade".to_string())
        .with_column(int_col("t_id", true))
        .with_column(int_col("t_ca_id", false))
        .with_column(int_col("t_s_symb", false))
        .with_column(varchar_col("t_st_id", 10))
        .with_column(decimal_col("t_bid_price"))
        .with_column(decimal_col("t_trade_price"))
        .with_column(int_col("t_qty", false))
        .with_column(decimal_col("t_chrg"))
        .with_column(decimal_col("t_comm"))
        .with_column(ts_col("t_dts"))
        .with_column(int_col("t_is_cash", false));
    schema.add_table(trade);

    // TRADE_HISTORY
    let th_id = tid_alloc.allocate();
    let th = TableSchema::new(th_id, "trade_history".to_string())
        .with_column(int_col("th_t_id", true))
        .with_column(varchar_col("th_st_id", 10))
        .with_column(ts_col("th_dts"));
    schema.add_table(th);

    // HOLDING
    let hold_id = tid_alloc.allocate();
    let hold = TableSchema::new(hold_id, "holding".to_string())
        .with_column(int_col("h_t_id", true))
        .with_column(int_col("h_ca_id", false))
        .with_column(int_col("h_s_symb", false))
        .with_column(int_col("h_qty", false))
        .with_column(decimal_col("h_price"));
    schema.add_table(hold);

    // HOLDING_SUMMARY
    let hs_id = tid_alloc.allocate();
    let hs = TableSchema::new(hs_id, "holding_summary".to_string())
        .with_column(int_col("hs_ca_id", true))
        .with_column(int_col("hs_s_symb", true))
        .with_column(int_col("hs_qty", false));
    schema.add_table(hs);

    // LAST_TRADE
    let lt_id = tid_alloc.allocate();
    let lt = TableSchema::new(lt_id, "last_trade".to_string())
        .with_column(int_col("lt_s_symb", true))
        .with_column(decimal_col("lt_price"))
        .with_column(decimal_col("lt_vol"))
        .with_column(ts_col("lt_dts"));
    schema.add_table(lt);

    // MARKET_FEED (simplified ticker)
    let mf_id = tid_alloc.allocate();
    let mf = TableSchema::new(mf_id, "market_feed".to_string())
        .with_column(int_col("mf_s_symb", true))
        .with_column(decimal_col("mf_price"))
        .with_column(int_col("mf_vol", false))
        .with_column(ts_col("mf_dts"));
    schema.add_table(mf);

    schema
}

// ---------------------------------------------------------------------------
// Transaction templates
// ---------------------------------------------------------------------------

/// Trade-Order transaction: places a new buy/sell order.
pub fn trade_order_txn(
    txn_id: TransactionId,
    acct_id: i64,
    symbol: i64,
    qty: i64,
    bid_price: i64,
    trade_id: i64,
    level: IsolationLevel,
) -> IrTransaction {
    let mut stmts = Vec::new();

    // 1. Look up customer account
    stmts.push(sel("account", &["ca_bal", "ca_b_id", "ca_c_id"],
        Predicate::eq("ca_id", Value::Integer(acct_id))));

    // 2. Look up customer tier
    stmts.push(sel("customer", &["c_tier"],
        Predicate::eq("c_id", Value::Integer(acct_id))));

    // 3. Get security info
    stmts.push(sel("security", &["s_name", "s_ex_id", "s_num_out"],
        Predicate::eq("s_symb", Value::Integer(symbol))));

    // 4. Get last trade price
    stmts.push(sel("last_trade", &["lt_price", "lt_vol"],
        Predicate::eq("lt_s_symb", Value::Integer(symbol))));

    // 5. Look up current holdings
    stmts.push(sel("holding_summary", &["hs_qty"],
        Predicate::And(vec![
            Predicate::eq("hs_ca_id", Value::Integer(acct_id)),
            Predicate::eq("hs_s_symb", Value::Integer(symbol)),
        ])));

    // 6. Insert trade record
    stmts.push(ins("trade",
        &["t_id", "t_ca_id", "t_s_symb", "t_st_id", "t_bid_price", "t_qty", "t_is_cash"],
        vec![lit(trade_id), lit(acct_id), lit(symbol),
             IrExpr::Literal(Value::Text("PNDG".to_string())),
             lit(bid_price), lit(qty), lit(1)],
    ));

    // 7. Insert trade history
    stmts.push(ins("trade_history",
        &["th_t_id", "th_st_id"],
        vec![lit(trade_id), IrExpr::Literal(Value::Text("PNDG".to_string()))],
    ));

    IrTransaction {
        id: txn_id,
        label: format!("TradeOrder_a{}_s{}", acct_id, symbol),
        isolation_level: level,
        statements: stmts,
        read_only: false,
    }
}

/// Trade-Result transaction: records the outcome of a completed trade.
pub fn trade_result_txn(
    txn_id: TransactionId,
    trade_id: i64,
    acct_id: i64,
    symbol: i64,
    trade_price: i64,
    qty: i64,
    commission: i64,
    level: IsolationLevel,
) -> IrTransaction {
    let mut stmts = Vec::new();

    // 1. Read the pending trade
    stmts.push(sel("trade",
        &["t_ca_id", "t_s_symb", "t_qty", "t_bid_price"],
        Predicate::eq("t_id", Value::Integer(trade_id))));

    // 2. Update trade with result
    stmts.push(upd("trade",
        &[("t_trade_price", lit(trade_price)),
          ("t_chrg", lit(commission / 10)),
          ("t_comm", lit(commission)),
          ("t_st_id", IrExpr::Literal(Value::Text("CMPT".to_string())))],
        Predicate::eq("t_id", Value::Integer(trade_id))));

    // 3. Insert trade history completion
    stmts.push(ins("trade_history",
        &["th_t_id", "th_st_id"],
        vec![lit(trade_id), IrExpr::Literal(Value::Text("CMPT".to_string()))],
    ));

    // 4. Update holding summary
    stmts.push(upd("holding_summary",
        &[("hs_qty", lit(qty))],
        Predicate::And(vec![
            Predicate::eq("hs_ca_id", Value::Integer(acct_id)),
            Predicate::eq("hs_s_symb", Value::Integer(symbol)),
        ])));

    // 5. Insert holding
    stmts.push(ins("holding",
        &["h_t_id", "h_ca_id", "h_s_symb", "h_qty", "h_price"],
        vec![lit(trade_id), lit(acct_id), lit(symbol), lit(qty), lit(trade_price)],
    ));

    // 6. Update account balance
    stmts.push(upd("account",
        &[("ca_bal", lit(-trade_price * qty - commission))],
        Predicate::eq("ca_id", Value::Integer(acct_id))));

    // 7. Update broker stats
    stmts.push(upd("broker",
        &[("b_num_trades", lit(1)), ("b_comm_total", lit(commission))],
        Predicate::eq("b_id", Value::Integer(1))));

    IrTransaction {
        id: txn_id,
        label: format!("TradeResult_t{}", trade_id),
        isolation_level: level,
        statements: stmts,
        read_only: false,
    }
}

/// Market-Feed transaction: updates security prices from the market data feed.
pub fn market_feed_txn(
    txn_id: TransactionId,
    updates: &[(i64, i64, i64)], // (symbol, new_price, volume)
    level: IsolationLevel,
) -> IrTransaction {
    let mut stmts = Vec::new();

    for &(symbol, price, volume) in updates {
        // Update last-trade price
        stmts.push(upd("last_trade",
            &[("lt_price", lit(price)), ("lt_vol", lit(volume))],
            Predicate::eq("lt_s_symb", Value::Integer(symbol))));

        // Read triggered pending trades at this price
        stmts.push(IrStatement::Select(IrSelect {
            table: "trade".to_string(),
            columns: vec!["t_id".to_string(), "t_ca_id".to_string(), "t_qty".to_string()],
            predicate: Predicate::And(vec![
                Predicate::eq("t_s_symb", Value::Integer(symbol)),
                Predicate::eq("t_st_id", Value::Text("PNDG".to_string())),
                Predicate::le("t_bid_price", Value::Integer(price)),
            ]),
            for_update: false,
            for_share: false,
        }));

        // Insert market feed record
        stmts.push(ins("market_feed",
            &["mf_s_symb", "mf_price", "mf_vol"],
            vec![lit(symbol), lit(price), lit(volume)],
        ));
    }

    IrTransaction {
        id: txn_id,
        label: format!("MarketFeed_{}items", updates.len()),
        isolation_level: level,
        statements: stmts,
        read_only: false,
    }
}

// ---------------------------------------------------------------------------
// Program builders
// ---------------------------------------------------------------------------

/// Build a TPC-E IR program with all three transaction types.
pub fn tpce_program(level: IsolationLevel) -> IrProgram {
    let mut txn_alloc: IdAllocator<TransactionId> = IdAllocator::new();

    let t1 = txn_alloc.allocate();
    let t2 = txn_alloc.allocate();
    let t3 = txn_alloc.allocate();

    let txns = vec![
        trade_order_txn(t1, 100, 42, 50, 2500, 9001, level),
        trade_result_txn(t2, 9001, 100, 42, 2550, 50, 100, level),
        market_feed_txn(t3, &[(42, 2560, 1000), (43, 1800, 500)], level),
    ];

    IrProgram {
        id: WorkloadId::new(2),
        name: "tpce_standard".to_string(),
        transactions: txns,
        schema_name: "tpce".to_string(),
        metadata: HashMap::new(),
    }
}

/// Build a TPC-E workload with schema.
pub fn tpce_workload(level: IsolationLevel) -> Workload {
    let program = tpce_program(level);
    let schema = tpce_schema();
    Workload {
        id: program.id,
        name: program.name.clone(),
        program,
        schema,
        parameters: WorkloadParameters {
            transaction_bound: 3,
            operation_bound: 15,
            data_item_bound: 30,
            repetitions: 1,
        },
        annotations: HashMap::new(),
    }
}

/// Build a concurrent Trade-Order + Trade-Result scenario.
pub fn tpce_concurrent_trade(level: IsolationLevel) -> IrProgram {
    let mut txn_alloc: IdAllocator<TransactionId> = IdAllocator::new();

    let t1 = txn_alloc.allocate();
    let t2 = txn_alloc.allocate();
    let t3 = txn_alloc.allocate();

    let txns = vec![
        trade_order_txn(t1, 100, 42, 30, 2500, 9010, level),
        trade_order_txn(t2, 100, 42, 20, 2480, 9011, level),
        trade_result_txn(t3, 9010, 100, 42, 2520, 30, 80, level),
    ];

    IrProgram {
        id: WorkloadId::new(3),
        name: "tpce_concurrent_trade".to_string(),
        transactions: txns,
        schema_name: "tpce".to_string(),
        metadata: HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// Statement helpers
// ---------------------------------------------------------------------------

fn sel(table: &str, cols: &[&str], pred: Predicate) -> IrStatement {
    IrStatement::Select(IrSelect {
        table: table.to_string(),
        columns: cols.iter().map(|c| c.to_string()).collect(),
        predicate: pred,
        for_update: false,
        for_share: false,
    })
}

fn upd(table: &str, assignments: &[(&str, IrExpr)], pred: Predicate) -> IrStatement {
    IrStatement::Update(IrUpdate {
        table: table.to_string(),
        assignments: assignments.iter().map(|(c, e)| (c.to_string(), e.clone())).collect(),
        predicate: pred,
    })
}

fn ins(table: &str, cols: &[&str], vals: Vec<IrExpr>) -> IrStatement {
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
    fn test_tpce_schema_tables() {
        let schema = tpce_schema();
        let names = schema.table_names();
        assert!(names.contains(&"account".to_string()));
        assert!(names.contains(&"customer".to_string()));
        assert!(names.contains(&"broker".to_string()));
        assert!(names.contains(&"security".to_string()));
        assert!(names.contains(&"trade".to_string()));
        assert!(names.contains(&"trade_history".to_string()));
        assert!(names.contains(&"holding".to_string()));
        assert!(names.contains(&"holding_summary".to_string()));
        assert!(names.contains(&"last_trade".to_string()));
        assert!(names.contains(&"market_feed".to_string()));
        assert_eq!(names.len(), 10);
    }

    #[test]
    fn test_trade_order_txn() {
        let txn = trade_order_txn(
            TransactionId::new(1), 100, 42, 50, 2500, 9001,
            IsolationLevel::Serializable,
        );
        assert!(!txn.read_only);
        assert!(txn.has_writes());
        assert_eq!(txn.statement_count(), 7);
    }

    #[test]
    fn test_trade_result_txn() {
        let txn = trade_result_txn(
            TransactionId::new(2), 9001, 100, 42, 2550, 50, 100,
            IsolationLevel::Snapshot,
        );
        assert!(!txn.read_only);
        assert!(txn.has_writes());
        assert_eq!(txn.statement_count(), 7);
    }

    #[test]
    fn test_market_feed_txn_single() {
        let txn = market_feed_txn(
            TransactionId::new(3),
            &[(42, 2560, 1000)],
            IsolationLevel::ReadCommitted,
        );
        assert!(!txn.read_only);
        // 3 stmts per symbol
        assert_eq!(txn.statement_count(), 3);
    }

    #[test]
    fn test_market_feed_txn_multi() {
        let txn = market_feed_txn(
            TransactionId::new(4),
            &[(42, 2560, 1000), (43, 1800, 500), (44, 900, 200)],
            IsolationLevel::ReadCommitted,
        );
        assert_eq!(txn.statement_count(), 9);
    }

    #[test]
    fn test_tpce_program() {
        let prog = tpce_program(IsolationLevel::Serializable);
        assert_eq!(prog.transaction_count(), 3);
        assert!(prog.total_statements() > 15);
    }

    #[test]
    fn test_tpce_workload() {
        let wl = tpce_workload(IsolationLevel::Snapshot);
        assert_eq!(wl.program.transaction_count(), 3);
        assert_eq!(wl.schema.table_names().len(), 10);
    }

    #[test]
    fn test_tpce_concurrent_trade() {
        let prog = tpce_concurrent_trade(IsolationLevel::RepeatableRead);
        assert_eq!(prog.transaction_count(), 3);
        // Two trade orders + one trade result
        assert!(prog.total_statements() > 18);
    }

    #[test]
    fn test_account_table_columns() {
        let schema = tpce_schema();
        let acc = schema.get_table("account").unwrap();
        let cols = acc.column_names();
        assert!(cols.contains(&"ca_id".to_string()));
        assert!(cols.contains(&"ca_bal".to_string()));
    }

    #[test]
    fn test_trade_table_columns() {
        let schema = tpce_schema();
        let t = schema.get_table("trade").unwrap();
        let cols = t.column_names();
        assert!(cols.contains(&"t_id".to_string()));
        assert!(cols.contains(&"t_trade_price".to_string()));
        assert!(cols.contains(&"t_comm".to_string()));
    }
}
