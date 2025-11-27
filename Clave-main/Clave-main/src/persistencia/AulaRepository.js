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
}

module.exports = AulaRepository;