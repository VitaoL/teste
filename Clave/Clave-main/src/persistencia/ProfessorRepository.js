const Database = require('./database');
const Professor = require('../dominio/Professor');

class ProfessorRepository {
    async salvar(professor) {
        const pool = Database.getPool();
        const sql = `
            INSERT INTO professor (nome, email, senha_hash, valor_base_aula, biografia, foto_url, link_video_demo)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
        `;
        const params = [
            professor.obterNome(),
            professor.obterEmail(),
            professor.obterSenhaHash ? professor.obterSenhaHash() : null,
            professor.obterPrecoHora(),
            null,
            null,
            null,
        ];

        const { rows } = await pool.query(sql, params);
        return rows[0].id;
    }

    async buscarTodos() {
        const pool = Database.getPool();
        const sql = `
            SELECT p.id,
                   p.nome,
                   p.email,
                   COALESCE(string_agg(DISTINCT i.nome, ', '), 'Instrumento') AS instrumento,
                   p.valor_base_aula AS preco_hora
              FROM professor p
              LEFT JOIN professor_instrumento pi ON pi.professor_id = p.id
              LEFT JOIN instrumento i ON i.id = pi.instrumento_id
             GROUP BY p.id, p.nome, p.email, p.valor_base_aula
        `;
        const { rows } = await pool.query(sql);
        return rows.map(row => new Professor(row.id, row.nome, row.email, row.instrumento || 'Instrumento', Number(row.preco_hora || 0)));
    }

    async buscarPorId(id) {
        const pool = Database.getPool();
        const sql = `
            SELECT p.id,
                   p.nome,
                   p.email,
                   COALESCE(string_agg(DISTINCT i.nome, ', '), 'Instrumento') AS instrumento,
                   p.valor_base_aula AS preco_hora
              FROM professor p
              LEFT JOIN professor_instrumento pi ON pi.professor_id = p.id
              LEFT JOIN instrumento i ON i.id = pi.instrumento_id
             WHERE p.id = $1
             GROUP BY p.id, p.nome, p.email, p.valor_base_aula
        `;
        const { rows } = await pool.query(sql, [id]);

        if (rows.length === 0) {
            return null;
        }

        const row = rows[0];
        return new Professor(row.id, row.nome, row.email, row.instrumento || 'Instrumento', Number(row.preco_hora || 0));
    }

    async buscarPorEmail(email) {
        const pool = Database.getPool();
        const sql = `
            SELECT p.id,
                   p.nome,
                   p.email,
                   COALESCE(string_agg(DISTINCT i.nome, ', '), 'Instrumento') AS instrumento,
                   p.valor_base_aula AS preco_hora,
                   p.senha_hash
              FROM professor p
              LEFT JOIN professor_instrumento pi ON pi.professor_id = p.id
              LEFT JOIN instrumento i ON i.id = pi.instrumento_id
             WHERE p.email = $1
             GROUP BY p.id, p.nome, p.email, p.valor_base_aula, p.senha_hash
        `;
        const { rows } = await pool.query(sql, [email]);
        if (rows.length === 0) {
            return null;
        }
        const row = rows[0];
        return new Professor(row.id, row.nome, row.email, row.instrumento || 'Instrumento', Number(row.preco_hora || 0), row.senha_hash || null);
    }

    async atualizarNome(id, nome) {
        const pool = Database.getPool();
        const sql = `
            UPDATE professor
               SET nome = $1
             WHERE id = $2
         RETURNING id, nome, email, valor_base_aula AS preco_hora
        `;
        const { rows } = await pool.query(sql, [nome, id]);

        if (rows.length === 0) {
            return null;
        }

        const row = rows[0];
        return new Professor(row.id, row.nome, row.email, 'Instrumento', Number(row.preco_hora || 0));
    }
}

module.exports = ProfessorRepository;
