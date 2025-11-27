const Database = require('./database');
const Professor = require('../dominio/Professor');

class ProfessorRepository {
    async salvar(professor) {
        const pool = Database.getPool();
        const sql = 'INSERT INTO professor (nome, email, instrumento, preco_hora) VALUES ($1, $2, $3, $4) RETURNING id';
        const params = [
            professor.obterNome(),
            professor.obterEmail(),
            professor.obterInstrumento(),
            professor.obterPrecoHora(),
        ];

        const { rows } = await pool.query(sql, params);
        return rows[0].id;
    }

    async buscarPorId(id) {
        const pool = Database.getPool();
        const sql = 'SELECT id, nome, email, instrumento, preco_hora FROM professor WHERE id = $1';
        const { rows } = await pool.query(sql, [id]);

        if (rows.length === 0) {
            return null;
        }

        const row = rows[0];
        return new Professor(row.id, row.nome, row.email, row.instrumento, Number(row.preco_hora));
    }
}

module.exports = ProfessorRepository;
