const Database = require('./database');

class AulaRepository {
    async salvar(aula) {
        const pool = Database.getPool();
        // Ajustado para o novo esquema de banco (campos aluno_id, professor_id, etc)
        const sql = `INSERT INTO aula (aluno_id, professor_id, data_hora, status_id, modalidade, valor_acordado)
                     VALUES ($1, $2, $3, $4, $5, $6) RETURNING id`;
        
        const values = [
            aula.alunoId,
            aula.professorId,
            aula.dataHora,
            1, // ID 1 = SOLICITADA (Conforme insert_data.sql)
            'online',
            aula.valor
        ];
        
        const resultado = await pool.query(sql, values);
        return resultado.rows[0].id;
    }

    async listarPorAluno(alunoId) {
        const pool = Database.getPool();
        const sql = `
            SELECT a.id,
                   a.data_hora,
                   a.modalidade,
                   a.valor_acordado,
                   s.nome as status,
                   p.id   as professor_id,
                   p.nome as professor_nome,
                   COALESCE(inst.instrumento, 'Instrumento') as instrumento,
                    p.foto_url
              FROM aula a
              LEFT JOIN professor p ON p.id = a.professor_id
              LEFT JOIN status_aula s ON s.id = a.status_id
              LEFT JOIN LATERAL (
                    SELECT string_agg(DISTINCT i.nome, ', ') as instrumento
                      FROM professor_instrumento pi
                      LEFT JOIN instrumento i ON i.id = pi.instrumento_id
                     WHERE pi.professor_id = p.id
              ) inst ON true
             WHERE a.aluno_id = $1
             ORDER BY a.data_hora DESC;
        `;

        const { rows } = await pool.query(sql, [alunoId]);
        return rows;
    }

    async listarPorProfessor(professorId) {
        const pool = Database.getPool();
        const sql = `
            SELECT a.id,
                   a.data_hora,
                   a.modalidade,
                   a.valor_acordado,
                   s.nome as status,
                   al.id   as aluno_id,
                   al.nome as aluno_nome,
                   al.email as aluno_email
              FROM aula a
              LEFT JOIN aluno al ON al.id = a.aluno_id
              LEFT JOIN status_aula s ON s.id = a.status_id
             WHERE a.professor_id = $1
             ORDER BY a.data_hora DESC;
        `;

        const { rows } = await pool.query(sql, [professorId]);
        return rows;
    }
}

module.exports = AulaRepository;