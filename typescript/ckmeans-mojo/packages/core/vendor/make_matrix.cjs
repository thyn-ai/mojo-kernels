/**
 * Vendored from simple-statistics v7.12.0 (src/make_matrix.js), ISC License,
 * Copyright (c) 2014, Tom MacWright — see NOTICE for the full license text.
 * Mechanically converted from ESM to CommonJS for the fallback backend;
 * the logic is unchanged.
 */
/**
 * Create a new column x row matrix.
 *
 * @private
 * @param {number} columns
 * @param {number} rows
 * @return {Array<Array<number>>} matrix
 * @example
 * makeMatrix(10, 10);
 */
function makeMatrix(columns, rows) {
    const matrix = [];
    for (let i = 0; i < columns; i++) {
        const column = [];
        for (let j = 0; j < rows; j++) {
            column.push(0);
        }
        matrix.push(column);
    }
    return matrix;
}

module.exports = makeMatrix;
